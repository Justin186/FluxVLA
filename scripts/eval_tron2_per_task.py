#!/usr/bin/env python
# =============================================================================
#  TRON2 LoRA —— 离线评估：按任务 / 按动作维度分解 flow-matching loss
#
#  为什么需要：
#    训练时日志里只有一个「整批混合」的 loss（见 docs/tron2_multi_task_dataset_guide.md
#    §9.7~9.9）。当 loss 平台时，无法区分「收敛到 aleatoric floor」还是「过拟合」。
#    这个脚本用某个 checkpoint 离线跑一遍验证集，给出：
#      - 总体 loss
#      - 每个任务（Press red / black / green …）各自的 loss
#      - 每个动作维度各自贡献的 loss（定位是不是某个维度把总 loss 钉住了）
#
#  用法：
#    python scripts/eval_tron2_per_task.py                    # 用最新 checkpoint
#    python scripts/eval_tron2_per_task.py --ckpt <adapter.safetensors>
#    python scripts/eval_tron2_per_task.py --per-task 256      # 每任务采样数
#
#  说明：
#    - 复用训练用的 config（transforms / 统计 / tokenizer 完全一致）
#    - 只做前向，不反传；显存占用远小于训练
#    - 任务标签取自样本自带的 task_description，不依赖目录结构
# =============================================================================
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

FV = '/home/lab/tron_ws/FluxVLA'
DEFAULT_CFG = os.path.join(
    FV, 'configs/pi05/pi05_paligemma_tron2_cabinet_lora.py')
DEFAULT_WORK = os.path.join(FV, 'work_dirs/tron2_buttons_v2')

# 动作维度语义（见 config 里的 RelativeActions mask 注释）
DIM_GROUPS = [
    ('左臂 0-6', list(range(0, 7))),
    ('左夹爪 7', [7]),
    ('右臂 8-14', list(range(8, 15))),
    ('右夹爪 15', [15]),
]


def load_cfg(path):
    spec = importlib.util.spec_from_file_location('tron2_cfg', path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, FV)
    spec.loader.exec_module(mod)
    return mod


def build_model(cfg, adapter_path, device):
    """复刻 DDPTrainRunner.run_setup 的构建顺序（见 ddp_train_runner.py:319-341）。"""
    from fluxvla.engines import build_vla_from_cfg
    from fluxvla.engines.runners.ddp_train_runner import \
        _resolve_lora_target_modules
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    m = cfg.model
    vla = build_vla_from_cfg(m)
    vla.freeze_backbones()
    vla.from_pretrained()

    target_modules = _resolve_lora_target_modules(vla, m['lora_target_modules'])
    lora_config = LoraConfig(
        r=m['lora_rank'],
        lora_alpha=m.get('lora_alpha', m['lora_rank']),
        lora_dropout=m.get('lora_dropout', 0.0),
        target_modules=target_modules,
        modules_to_save=m.get('modules_to_save'),
        init_lora_weights='gaussian')
    vla = get_peft_model(vla, lora_config)

    result = set_peft_model_state_dict(vla, load_file(adapter_path))
    missing = list(getattr(result, 'missing_keys', None) or [])
    unexpected = list(getattr(result, 'unexpected_keys', None) or [])
    print(f'[ok] adapter: {os.path.basename(adapter_path)} '
          f'(missing={len(missing)}, unexpected={len(unexpected)})')
    if missing or unexpected:
        print(f'     missing[:5]={missing[:5]}')
        print(f'     unexpected[:5]={unexpected[:5]}')

    vla = vla.to(device=device, dtype=torch.bfloat16)
    vla.eval()
    return vla


def build_dataset(cfg, data_roots=None):
    """data_roots 非空时替换 data_root_path（用于在留出数据上评估泛化）。"""
    import copy

    from fluxvla.engines import build_dataset_from_cfg
    wc = cfg.train_dataloader
    inner = copy.deepcopy(wc['dataset']['datasets'][0])
    if data_roots:
        inner['data_root_path'] = list(data_roots)
    ds = build_dataset_from_cfg(inner)
    stats = json.load(
        open(wc['dataset']['dataset_statistics_path'], encoding='utf-8'))
    return ds, stats


def build_prompt_processor(cfg):
    """取出 dataset transform 里的 ProcessPrompts，用于替换 prompt 后重新分词。"""
    from fluxvla.engines import build_transform_from_cfg
    inner = cfg.train_dataloader['dataset']['datasets'][0]
    for t in inner['transforms']:
        if t.get('type') == 'ProcessPrompts':
            return build_transform_from_cfg(t)
    return None


def make_batches(ds, stats, per_task, batch_size, seed=0,
                 pp=None, prompt_override=None):
    """按 root 均匀采样索引，再按样本真实的 task_description 分组。

    prompt_override 非空时，把 prompt 里的任务文本换成该字符串再重新分词，
    用于复现「训练 prompt vs 部署 prompt 不一致」的影响。
    """
    cum = np.asarray(ds.dataset_cumulative_sizes)
    rng = np.random.default_rng(seed)
    idxs = []
    for i in range(len(cum) - 1):
        lo, hi = int(cum[i]) + 2, int(cum[i + 1]) - 2
        n = min(per_task, hi - lo)
        idxs.extend(rng.choice(np.arange(lo, hi), size=n, replace=False).tolist())
    rng.shuffle(idxs)

    samples = defaultdict(list)      # task -> list of samples
    for i in idxs:
        try:
            s = ds.__getitem__(int(i), stats)
        except Exception as e:  # noqa: BLE001
            print(f'  [warn] index {i} 读取失败: {type(e).__name__}: {e}')
            continue
        if prompt_override and pp is not None:
            head, sep, tail = str(s['prompt']).partition(', State:')
            s = dict(s)
            s['prompt'] = f'Task: {prompt_override}{sep}{tail}'
            s = pp(s)
        samples[s['task_description']].append(s)

    batches = []
    for task, items in sorted(samples.items()):
        for k in range(0, len(items), batch_size):
            batches.append((task, items[k:k + batch_size]))
    return batches, {t: len(v) for t, v in sorted(samples.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=DEFAULT_CFG)
    ap.add_argument('--work-dir', default=DEFAULT_WORK)
    ap.add_argument('--ckpt', default=None,
                    help='LoRA adapter 路径；默认用 work_dir/adapter_model.safetensors')
    ap.add_argument('--per-task', type=int, default=256, help='每个数据集采样多少个样本')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default=None, help='把结果写成 JSON')
    ap.add_argument('--data-roots', default=None,
                    help='逗号分隔，替换 data_root_path（在留出数据上评估泛化）')
    ap.add_argument('--prompt-override', default=None,
                    help='把所有样本的 prompt 任务文本换成该字符串（复现 prompt 不一致）')
    args = ap.parse_args()
    data_roots = ([s.strip() for s in args.data_roots.split(',')]
                  if args.data_roots else None)

    os.chdir(FV)
    adapter_path = args.ckpt or os.path.join(
        args.work_dir, 'adapter_model.safetensors')
    if not os.path.isfile(adapter_path):
        sys.exit(f'[FAIL] 找不到 adapter: {adapter_path}')

    print('=' * 74)
    print(f'adapter   : {adapter_path}')
    print(f'config    : {args.config}')
    print(f'per-task  : {args.per_task}  batch={args.batch_size}')
    print('=' * 74)

    cfg = load_cfg(args.config)
    device = torch.device(args.device)

    print('[1/3] 构建数据集 ...')
    ds, stats = build_dataset(cfg, data_roots)
    print(f'      len={len(ds)}  window={ds.action_window_size}  '
          f'roots={list(ds.dataset_cumulative_sizes)}')
    print(f'      tasks={[t[0] for t in ds.tasks]}')
    if data_roots:
        print(f'      [留出集] data_root_path 已替换为 {len(data_roots)} 个目录')

    print('[2/3] 加载模型 + LoRA adapter ...')
    model = build_model(cfg, adapter_path, device)

    print('[3/3] 采样 + 前向 ...')
    pp = build_prompt_processor(cfg) if args.prompt_override else None
    if args.prompt_override:
        print(f'      [prompt 替换] 所有样本的任务文本 -> {args.prompt_override!r}')
        if pp is None:
            sys.exit('[FAIL] 找不到 ProcessPrompts，无法重新分词')
    batches, per_task_n = make_batches(
        ds, stats, args.per_task, args.batch_size,
        pp=pp, prompt_override=args.prompt_override)
    print(f'      采样到: {per_task_n}')

    # 在 reduce 之前截住未归约的逐元素 loss
    import fluxvla.models.vlas.pi0_flowmatching as fm
    original_reduce = fm.reduce_action_bc_loss
    captured = {}

    def spy(losses, action_mask=None, sample_weight=None, **kw):
        captured['losses'] = losses.detach().float().cpu()
        captured['mask'] = (None if action_mask is None
                            else action_mask.detach().float().cpu())
        return original_reduce(
            losses, action_mask=action_mask, sample_weight=sample_weight, **kw)

    fm.reduce_action_bc_loss = spy
    tensor_keys = ['images', 'lang_tokens', 'states', 'actions',
                   'action_masks', 'img_masks', 'lang_masks']

    # 累加器：key -> [sum(loss), count]
    acc = defaultdict(lambda: [0.0, 0.0])
    per_dim_sum = None
    per_dim_cnt = None
    done = 0

    try:
        with torch.no_grad():
            for task, items in batches:
                batch = {
                    k: torch.stack([torch.as_tensor(s[k]) for s in items]).to(device)
                    for k in tensor_keys
                }
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(**batch)
                losses = captured['losses']            # [B, T, D]
                mask = captured['mask']                # [B, T] 或 [B, T, D]
                if mask is None:
                    mask = torch.ones_like(losses)
                elif mask.dim() == 2:
                    mask = mask.unsqueeze(-1).expand_as(losses)
                elif mask.dim() == 3 and mask.shape[-1] == 1:
                    mask = mask.expand_as(losses)

                s_all, c_all = float((losses * mask).sum()), float(mask.sum())
                acc['__all__'][0] += s_all
                acc['__all__'][1] += c_all
                acc[task][0] += s_all
                acc[task][1] += c_all

                ds_sum = (losses * mask).sum(dim=(0, 1))
                ds_cnt = mask.sum(dim=(0, 1))
                if per_dim_sum is None:
                    per_dim_sum = ds_sum.clone()
                    per_dim_cnt = ds_cnt.clone()
                else:
                    per_dim_sum += ds_sum
                    per_dim_cnt += ds_cnt

                done += len(items)
                if done % 128 == 0 or done == sum(per_task_n.values()):
                    cur = acc['__all__'][0] / max(acc['__all__'][1], 1)
                    print(f'      {done:5d} 样本  当前总体 loss = {cur:.5f}')
    finally:
        fm.reduce_action_bc_loss = original_reduce

    # ---------------- 汇总 ----------------
    print('\n' + '=' * 74)
    print('结果')
    print('=' * 74)
    total = acc['__all__'][0] / max(acc['__all__'][1], 1)
    print(f'\n总体验证 loss = {total:.5f}   (RMSE={total ** 0.5:.4f})')

    print('\n【按任务】')
    print(f'  {"任务":32s} {"loss":>9s} {"RMSE":>8s} {"样本权重大小":>14s}')
    task_res = {}
    for task in sorted(t for t in acc if t != '__all__'):
        s, c = acc[task]
        if c <= 0:
            continue
        v = s / c
        task_res[task] = v
        print(f'  {task:32s} {v:9.5f} {v ** 0.5:8.4f} {int(c):14d}')

    print('\n【按动作维度】')
    pd_cnt = (per_dim_cnt if per_dim_cnt is not None
              else torch.ones(1))
    pd_sum = per_dim_sum if per_dim_sum is not None else torch.zeros(1)
    pd_loss = (pd_sum / pd_cnt.clamp(min=1)).tolist()
    dim_res = {}
    print(f'  {"维度":12s} {"loss":>9s} {"占总数比例":>11s}')
    for name, dims in DIM_GROUPS:
        vals = [pd_loss[d] for d in dims if d < len(pd_loss)]
        if not vals:
            continue
        v = float(np.mean(vals))
        dim_res[name] = v
        share = v * len(vals) / max(len(pd_loss), 1) / max(total, 1e-9)
        print(f'  {name:12s} {v:9.5f} {100 * share:10.1f}%')
    print(f'\n  逐维明细: ' + ' '.join(f'{x:.4f}' for x in pd_loss))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as h:
            json.dump(dict(adapter=adapter_path, overall=total,
                           per_task=task_res, per_dim_group=dim_res,
                           per_dim=pd_loss, samples=per_task_n),
                      h, ensure_ascii=False, indent=2)
        print(f'\n[ok] 写入 {args.out}')


if __name__ == '__main__':
    main()
