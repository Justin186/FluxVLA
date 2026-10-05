#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以厂商验证过的 pi05_paligemma_tron2_full_finetune.py 为基底，
生成一份适配「数采软件导出的 5 个数据集 + 单卡 24G + LoRA」的训练 config。

只做定点替换，其余参数完全沿用厂商配置，避免手写抄错。

改动清单：
  1. ParquetDataset -> ParquetDatasetV3      （v3.0 数据集必须用 v3 加载器）
  2. data_root_path   -> 5 个新数据集
  3. 加入 LoRA（rank=64, alpha=128，target 模块照搬官方 LoRA 配置）
  4. per_device_batch_size 8 -> 1            （单卡 24G）
  5. grad_accumulation_steps 2 -> 8          （维持有效 batch）
  6. enable_gradient_checkpointing -> True   （省显存）
"""

import argparse
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'FluxVLA/configs/pi05/pi05_paligemma_tron2_full_finetune.py')
DATA_DIR = os.path.join(BASE, 'datasets/RealRobot_Tron2_lerobot')


def discover_datasets():
    """自动扫描数据集目录，返回按名排序的相对路径列表。

    以后新增数据集只要丢进 DATA_DIR 再重跑本脚本即可，不用手改 config。
    """
    if not os.path.isdir(DATA_DIR):
        print("  [FAIL] 数据集目录不存在: %s" % DATA_DIR)
        sys.exit(1)
    names = sorted(d for d in os.listdir(DATA_DIR)
                   if os.path.isfile(os.path.join(DATA_DIR, d, 'meta', 'info.json')))
    if not names:
        print("  [FAIL] %s 下没有可用数据集" % DATA_DIR)
        sys.exit(1)
    return ['./datasets/RealRobot_Tron2_lerobot/%s' % n for n in names]


DATASETS = discover_datasets()

LORA_BLOCK = """    use_lora=True,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.0,
    lora_target_modules=[
        'q_proj',
        'v_proj',
        'k_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
        'projector.projector',
        'out_proj',
        'fc1',
        'fc2',
    ],
    modules_to_save=[
        'action_in_proj',
        'action_out_proj',
        'time_mlp_in',
        'time_mlp_out',
    ],
"""


def rep_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        print("  [FAIL] %s: 锚点出现 %d 次（应为 1 次）" % (tag, n))
        sys.exit(1)
    print("  [ok]   %s" % tag)
    return text.replace(old, new)


def main():
    ap = argparse.ArgumentParser(
        description='生成 TRON2 的 pi05 LoRA 训练 config（数据集自动扫描）')
    ap.add_argument('--out', default='configs/pi05/pi05_paligemma_tron2_cabinet_lora.py',
                    help='输出 config 路径（相对 FluxVLA 目录）')
    ap.add_argument('--max-steps', type=int, default=20000,
                    help='总训练步数；续训时应设为「累计总步数」')
    ap.add_argument('--decay-steps', type=int, default=None,
                    help='cosine 衰减终点，默认等于 max_steps')
    ap.add_argument('--warmup-steps', type=int, default=1000)
    ap.add_argument('--loss-action-dim', type=int, default=16,
                    help='参与 flow-matching loss 的动作维度；我们只有 16 维真实动作')
    ap.add_argument('--save-iter-interval', type=int, default=None,
                    help='存档间隔，默认取 max_steps/2（保证至少有一次存档）')
    ap.add_argument('--batch-size', type=int, default=3,
                    help='单卡 micro-batch；4090D 24G 实测上限为 3（4 会 OOM）')
    ap.add_argument('--grad-accum', type=int, default=8,
                    help='梯度累积步数；有效 batch = batch-size * grad-accum')
    args = ap.parse_args()

    dst = os.path.join(BASE, 'FluxVLA', args.out)
    decay_steps = args.decay_steps or args.max_steps
    save_iter = args.save_iter_interval or max(1, args.max_steps // 2)

    with open(SRC, encoding='utf-8') as f:
        t = f.read()

    print("基底: %s" % os.path.relpath(SRC, BASE))
    print("数据集 %d 个:" % len(DATASETS))
    for d in DATASETS:
        print("   ", os.path.basename(d))
    print("max_steps=%d  decay_steps=%d  save_iter_interval=%d"
          % (args.max_steps, decay_steps, save_iter))

    # 1) 数据集类
    t = rep_once(t, "type='ParquetDataset'", "type='ParquetDatasetV3'",
                 "ParquetDataset -> ParquetDatasetV3")

    # 2) data_root_path
    m = re.search(r'data_root_path=\s*# noqa: E251\s*\n\s*\[\s*\n[^\]]*\],\n', t)
    if not m:
        print("  [FAIL] 找不到 data_root_path 块")
        sys.exit(1)
    new_paths = ("data_root_path=[  # noqa: E501\n"
                 + "".join("                '%s',  # noqa: E501\n" % d for d in DATASETS)
                 + "            ],\n")
    t = t[:m.start()] + new_paths + t[m.end():]
    print("  [ok]   data_root_path -> %d 个数据集" % len(DATASETS))

    # 3) LoRA
    t = rep_once(t, "    freeze_vision_backbone=False,\n",
                 "    freeze_vision_backbone=False,\n" + LORA_BLOCK,
                 "插入 LoRA 配置")

    # 4) 单卡 batch
    t = rep_once(t, "    per_device_batch_size=8,\n",
                 "    per_device_batch_size=%d,\n" % args.batch_size,
                 "per_device_batch_size 8 -> %d" % args.batch_size)

    # 5) 梯度累积
    t = rep_once(t, "    grad_accumulation_steps=2,\n",
                 "    grad_accumulation_steps=%d,\n" % args.grad_accum,
                 "grad_accumulation_steps 2 -> %d" % args.grad_accum)

    # 6) 梯度检查点
    t = rep_once(t, "    enable_gradient_checkpointing=False,\n",
                 "    enable_gradient_checkpointing=True,\n",
                 "enable_gradient_checkpointing -> True")

    # 7) 去掉 wandb（无 API key，只保留本地 jsonl 日志）
    t = rep_once(t, "active_trackers=('jsonl', 'wandb'),",
                 "active_trackers=('jsonl',),",
                 "active_trackers: 去掉 wandb")

    # 8) 关掉 EMA：EMA 会 clone 全部 named_parameters()（含冻结的 3.5B 骨干），
    #    单卡 24G 直接 OOM。官方 LoRA config 同样没开 EMA。
    t = rep_once(t, "    ema_decay=0.99,\n", "", "去掉 ema_decay（省 14G）")

    # 9) keep_params_fp32 必须保持 True：
    #    base_train_runner 里 batch_dtype = ... and not keep_params_fp32 else None，
    #    设成 False 会让 batch 变 bf16 而参数仍是 fp32 -> dtype 不匹配报错。
    #    （这是 OpenPI 的设计：fp32 主参 + autocast 做 bf16 计算）

    # 10) runner: FSDP -> DDP。LoRA 只在 DDPTrainRunner 里实现，
    #     FSDPTrainRunner 里完全没有 lora/peft 代码，use_lora 会被静默忽略。
    t = rep_once(t, "    type='FSDPTrainRunner',\n", "    type='DDPTrainRunner',\n",
                 "runner: FSDPTrainRunner -> DDPTrainRunner")

    # 11) 去掉 DDP 不支持的 FSDP 专属键
    #     （DDPTrainRunner 结尾对未知字段会主动抛 TypeError）
    t = rep_once(t, "    sharding_strategy='global-shard-grad-op',\n", "",
                 "去掉 sharding_strategy（FSDP 专属）")
    t = rep_once(t, "    fsdp_wrap_policy='execution-block',\n", "",
                 "去掉 fsdp_wrap_policy（FSDP 专属）")
    t = rep_once(t, "    keep_params_fp32=True,\n    change_key_name=False)",
                 "    keep_params_fp32=True)",
                 "去掉 change_key_name（DDP 不支持）")

    # 12) DDPTrainRunner 的 max_epochs 默认是 10（FSDP 的是 None），
    #     于是 max_steps 和 max_epochs 会同时被设上 -> 断言失败。显式置 None。
    t = rep_once(t, "    type='DDPTrainRunner',\n    max_steps=20_000,\n",
                 "    type='DDPTrainRunner',\n    max_epochs=None,\n"
                 "    max_steps=20_000,\n",
                 "max_epochs 显式置 None")

    # 13) 用 bf16 权重 + keep_params_fp32=False。
    #     keep_params_fp32=True 会同时持有 bf16 计算副本和 fp32 主参
    #     (3.69B: 7.4G + 14.8G ≈ 22G)，单卡 24G 必然 OOM。
    #     换成 bf16 权重后 batch 与参数同为 bf16，不再 dtype 冲突。
    t = rep_once(t, "'./checkpoints/pi05_base/model.safetensors'",
                 "'./checkpoints/pi05_base_bf16/model.safetensors'",
                 "pretrained_name_or_path -> bf16 权重")
    t = rep_once(t, "    keep_params_fp32=True)", "    keep_params_fp32=False)",
                 "keep_params_fp32 -> False（配 bf16 权重）")

    # 14) openpi_fp32_flow 默认是 False，官方 LoRA config 没设。
    #     TRON2 全参 config 设了 True，会让 action_in_proj 收到 fp32 输入
    #     而 bf16 权重 -> mat1/mat2 dtype 不一致。LoRA 路线必须关掉。
    t = rep_once(t, "    openpi_fp32_flow=True,\n", "    openpi_fp32_flow=False,\n",
                 "openpi_fp32_flow -> False")

    # 17) loss_action_dim 32 -> 16：只有前 16 维是真实动作，dims 16-31 是
    #     NormalizeStatesAndActions 补的 0，在 flow matching 里 t->0 时不可学，
    #     会拉高 loss 地板并浪费动作头容量。
    t = rep_once(t, "    loss_action_dim=32,\n",
                 "    loss_action_dim=%d,\n" % args.loss_action_dim,
                 "loss_action_dim -> %d" % args.loss_action_dim)

    # 15) static_graph=False。DDPTrainRunner 里 find_unused_parameters 是硬编码 True，
    #     而 static_graph 默认 True，两者叠加 + LoRA 大量未用参数 + 梯度检查点
    #     会触发 expect_autograd_hooks_ INTERNAL ASSERT FAILED。
    t = rep_once(t, "    keep_params_fp32=False)",
                 "    keep_params_fp32=False,\n    static_graph=False)",
                 "static_graph -> False")

    # 16) 训练步数 + LR schedule（续训时按「累计总时间线」设定）
    t = rep_once(t, "    max_steps=20_000,\n",
                 "    max_steps=%d,\n    save_iter_interval=%d,\n"
                 % (args.max_steps, save_iter),
                 "max_steps=%d, save_iter_interval=%d" % (args.max_steps, save_iter))
    t = rep_once(t, "        warmup_steps=1000,",
                 "        warmup_steps=%d," % args.warmup_steps,
                 "warmup_steps=%d" % args.warmup_steps)
    t = rep_once(t, "        decay_steps=30000,",
                 "        decay_steps=%d," % decay_steps,
                 "decay_steps=%d" % decay_steps)

    with open(dst, 'w', encoding='utf-8') as f:
        f.write(t)
    print()
    print("已写出: %s" % os.path.relpath(dst, BASE))
    print()
    print("训练命令（torchrun，单卡）:")
    print("  cd %s/FluxVLA" % BASE)
    print("  HF_ENDPOINT=https://hf-mirror.com WANDB_MODE=disabled \\")
    print("  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\")
    print("  /opt/miniconda3/envs/fluxvla/bin/torchrun --standalone --nnodes 1 \\")
    print("      --nproc-per-node 1 scripts/train.py \\")
    print("      --config %s \\" % args.out)
    print("      --work-dir work_dirs/tron2_v1")
    print()
    print("续训时追加 --resume-from work_dirs/tron2_v1/checkpoints/latest-checkpoint.pt")
    print("并重新生成 config（--max-steps / --decay-steps 按累计总步数设定）。")


if __name__ == '__main__':
    main()
