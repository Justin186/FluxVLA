#!/usr/bin/env python
# =============================================================================
#  TRON2 端到端推理自测（离线，不接机器人）
#
#  干什么：
#    从真机数据集里取【真实观测】(3 路相机 mp4 帧 + parquet 状态)，按部署契约
#    打包成 observation，打 ZMQ 推理服务，拿回 (50, 18) 反归一化动作。
#
#  为什么要它：
#    1. 验证部署链路能不能跑通（Triton + CUDA Graph + delta 还原 -> 18 维）
#    2. 验证 prompt 路由：同一个观测换 task_description，输出必须不同
#    3. 量化「训练 prompt vs 部署 prompt」不一致的代价（P0）
#
#  观测契约（见 docs/tron2_cabinet_lora_rebuild_guide.md §6.5）：
#    qpos   float32[16]  策略顺序 [L7, gripL, R7, gripR]，走归一化
#    states float32[18]  机器人原生顺序 [L7, R7, head2, gripL, gripR]，做 delta 还原
#    cam_high / cam_left_wrist / cam_right_wrist   uint8[H, W, 3] RGB
#    task_description  str，必须与数据集 meta/tasks.parquet 逐字一致
#
#  用法：
#    scripts/zmq_inference_server.sh --config configs/pi05/\
#        pi05_paligemma_tron2_cabinet_lora_deploy.py --ckpt-path <*.safetensors> --port 5555
#    python scripts/test_tron2_inference_offline.py
#    python scripts/test_tron2_inference_offline.py --frame 60 --repeat 5
# =============================================================================
import argparse
import glob
import json
import os
import sys
import uuid

import numpy as np

FV = '/home/lab/tron_ws/FluxVLA'
DATA_ROOT = os.path.join(FV, 'datasets/RealRobot_Tron2_lerobot')
CAMS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

# 18 维机器人布局 [L7, R7, head2, gripL, gripR]
R_ARM = slice(7, 14)
GRIP_L, GRIP_R = 16, 17
HEAD = slice(14, 16)

WRONG_PROMPT = 'complete the task'
FPS_FALLBACK = 30.0
HORIZON = 50          # 与 n_action_steps 一致


# --------------------------------------------------------------------------- #
# 观测读取
# --------------------------------------------------------------------------- #
def load_episodes(ds_dir):
    import pandas as pd
    import pyarrow.parquet as pq

    files = sorted(
        glob.glob(os.path.join(ds_dir, 'meta', 'episodes', '**', '*.parquet'),
                  recursive=True))
    if not files:
        raise FileNotFoundError(f'找不到 episode 元数据: {ds_dir}')
    return pd.concat([pq.read_table(f).to_pandas() for f in files],
                     ignore_index=True)


def read_meta(ds_dir):
    import pyarrow.parquet as pq

    with open(os.path.join(ds_dir, 'meta', 'info.json'), encoding='utf-8') as h:
        info = json.load(h)
    fps = float(info.get('fps') or FPS_FALLBACK)
    tasks = pq.read_table(
        os.path.join(ds_dir, 'meta', 'tasks.parquet')).to_pandas()['task'].tolist()
    return fps, [str(t) for t in tasks]


def _mp4(ds_dir, cam, row):
    col = f'videos/observation.images.{cam}'
    return os.path.join(
        ds_dir, 'videos', f'observation.images.{cam}',
        f"chunk-{int(row[col + '/chunk_index']):03d}",
        f"file-{int(row[col + '/file_index']):03d}.mp4")


def read_state(ds_dir, row, k):
    """取该 episode 第 k 帧的 observation.state（16 维原始值）。"""
    import pyarrow.parquet as pq

    path = os.path.join(
        ds_dir, 'data', f"chunk-{int(row['data/chunk_index']):03d}",
        f"file-{int(row['data/file_index']):03d}.parquet")
    cols = pq.read_table(path).to_pandas()
    if 'episode_index' in cols.columns:
        sub = cols[cols['episode_index'] == int(row['episode_index'])]
        if len(sub):
            cols = sub
    state = np.asarray(cols.iloc[k]['observation.state'], dtype=np.float32)
    if state.shape != (16, ):
        raise ValueError(f'observation.state 应为 16 维，实际 {state.shape}')
    return state


def read_image(ds_dir, cam, row, k, fps):
    """用仓库自己的解码器取帧，保证与训练时同一路径（RGB, uint8）。"""
    from fluxvla.datasets.utils.video_decode import decode_video_frames

    col = f'videos/observation.images.{cam}'
    ts = float(row[col + '/from_timestamp']) + k / fps
    frame = decode_video_frames(_mp4(ds_dir, cam, row), [ts])[0]
    img = frame.numpy()
    if img.shape[0] == 3:                     # CHW -> HWC
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def read_actions(ds_dir, row, k, horizon):
    """读该 episode 第 k 帧起 horizon 步的【原始绝对动作】(T, 16)。"""
    import pyarrow.parquet as pq

    path = os.path.join(
        ds_dir, 'data', f"chunk-{int(row['data/chunk_index']):03d}",
        f"file-{int(row['data/file_index']):03d}.parquet")
    cols = pq.read_table(path).to_pandas()
    if 'episode_index' in cols.columns:
        sub = cols[cols['episode_index'] == int(row['episode_index'])]
        if len(sub):
            cols = sub
    block = cols.iloc[k:k + horizon]['action'].tolist()
    return np.stack([np.asarray(x, dtype=np.float32) for x in block])


def pred18_to_16(a18):
    """(T,18) [L7,R7,head2,gripL,gripR] -> (T,16) [L7,gripL,R7,gripR]。"""
    out = np.empty((a18.shape[0], 16), dtype=np.float32)
    out[:, 0:7] = a18[:, 0:7]       # L7
    out[:, 7] = a18[:, GRIP_L]      # gripL
    out[:, 8:15] = a18[:, R_ARM]    # R7
    out[:, 15] = a18[:, GRIP_R]     # gripR
    return out


def build_observation(ds_dir, row, k, fps, task):
    s16 = read_state(ds_dir, row, k)
    # [L7, gripL, R7, gripR] 就是策略顺序，直接喂 qpos
    qpos = np.ascontiguousarray(s16, dtype=np.float32)
    # 机器人原生 18 维: [L7, R7, head2, gripL, gripR]
    # 数据集没有头部，填 0 —— 头部是纯透传维度（docs §9.9 已验证）
    states = np.ascontiguousarray(
        np.concatenate([
            s16[0:7], s16[8:15],
            np.zeros(2, np.float32), s16[7:8], s16[15:16],
        ]), dtype=np.float32)
    obs = {cam: read_image(ds_dir, cam, row, k, fps) for cam in CAMS}
    obs.update(qpos=qpos, states=states, task_description=str(task))
    return obs


# --------------------------------------------------------------------------- #
# ZMQ
# --------------------------------------------------------------------------- #
def make_client(zmq_addr, timeout_ms=600_000):
    import zmq
    from fluxvla.engines.runners.serving.serializers import MsgSerializer

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(zmq_addr)
    return ctx, sock, MsgSerializer


def predict(sock, serializer, obs, unnorm_key='private', seed=7,
            episode_id='offline-test', reset=True):
    import time

    sock.send(serializer.to_bytes({
        'endpoint': 'predict_action',
        'data': {
            'observation': obs,
            'unnorm_key': unnorm_key,
            'episode_id': episode_id,
            'seed': seed,
            'reset': reset,
            'request_id': uuid.uuid4().hex,
        },
    }))
    t0 = time.perf_counter()
    resp = serializer.from_bytes(sock.recv())
    wall = (time.perf_counter() - t0) * 1000.0
    if not resp.get('ok', False):
        raise RuntimeError(f"服务端报错: {resp.get('error')}")
    return (np.asarray(resp['actions'], dtype=np.float32),
            float(resp.get('inference_time_s', 0.0)) * 1000.0, wall)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def summarise(a):
    return dict(
        shape=tuple(a.shape),
        r_arm=float(np.abs(a[:, R_ARM]).max()),
        grip_l=float(a[0, GRIP_L]),
        grip_r=float(a[0, GRIP_R]),
        head=a[0, HEAD].tolist(),
        finite=bool(np.isfinite(a).all()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zmq', default='tcp://127.0.0.1:5555')
    ap.add_argument('--frame', type=int, default=60,
                    help='每条 episode 取第几帧（避开前 10 帧黑帧）')
    ap.add_argument('--episode', type=int, default=0, help='取第几条 episode')
    ap.add_argument('--repeat', type=int, default=3, help='每个 prompt 重复几次测延迟')
    ap.add_argument('--unnorm-key', default='private')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    os.chdir(FV)
    sys.path.insert(0, FV)

    ds_dirs = sorted(
        d for d in glob.glob(os.path.join(DATA_ROOT, 'lerobot_*'))
        if os.path.isdir(d))
    if not ds_dirs:
        sys.exit(f'[FAIL] 找不到数据集: {DATA_ROOT}/lerobot_*')

    print('=' * 78)
    print(f'ZMQ 服务 : {args.zmq}')
    print(f'数据集   : {len(ds_dirs)} 个')
    print('=' * 78)

    ctx, sock, serializer = make_client(args.zmq)
    try:
        # ---------- 1. 构建观测 ----------
        jobs = []
        for ds_dir in ds_dirs:
            fps, tasks = read_meta(ds_dir)
            eps = load_episodes(ds_dir)
            row = eps[(eps['episode_index'] == args.episode)]
            if not len(row):
                row = eps.iloc[[0]]
            row = row.iloc[0]
            length = int(row['length'])
            k = min(args.frame, length - 2)
            task = tasks[0]
            print(f'\n[观测] {os.path.basename(ds_dir)}')
            print(f'       任务={task!r}  episode={int(row["episode_index"])} '
                  f'帧={k}/{length}  fps={fps}')
            obs = build_observation(ds_dir, row, k, fps, task)
            img = obs['cam_high']
            print(f'       cam_high {img.shape} {img.dtype} '
                  f'mean={img.mean():.1f}  '
                  f'qpos范围=[{obs["qpos"].min():+.3f}, {obs["qpos"].max():+.3f}]')
            jobs.append(dict(name=os.path.basename(ds_dir), task=task, obs=obs,
                             ds_dir=ds_dir, row=row, k=k, length=length))

        all_tasks = []
        for j in jobs:
            if j['task'] not in all_tasks:
                all_tasks.append(j['task'])

        # ---------- 2. warmup ----------
        print('\n[warmup] 第一次调用要跑 Triton JIT + CUDA Graph，可能几秒 ...')
        _, t_srv, t_wall = predict(sock, serializer, jobs[0]['obs'],
                                   unnorm_key=args.unnorm_key,
                                   seed=args.seed)
        print(f'         首次调用失败则为 0；本次 服务端={t_srv:.1f}ms '
              f'往返={t_wall:.1f}ms')

        # ---------- 3. 逐观测 × 逐 prompt ----------
        rows = []
        for job in jobs:
            name, task, obs = job['name'], job['task'], job['obs']
            print('\n' + '-' * 78)
            print(f'观测 {name}   标签任务 = {task!r}')
            print('-' * 78)
            header = (f'  {"prompt":38s} {"服务端ms":>9s} {"往返ms":>8s} '
                      f'{"max|R7|":>9s} {"gripL":>8s} {"gripR":>8s} '
                      f'{"vs正确":>8s} {"右臂MAE":>8s}')
            print(header)
            print('  ' + '-' * (len(header) - 2))

            # 数据集真值（原始绝对关节角），用来给每种 prompt 算误差
            # ★ DenormalizeDeltaAction 已把 delta 还原成【绝对关节角】
            #   （normalize.py:507 `action[..., :dims] += state`），可直接比。
            gt_abs = None
            try:
                gt_abs = read_actions(job['ds_dir'], job['row'], job['k'],
                                      HORIZON)
            except Exception as exc:  # noqa: BLE001
                print(f'  [真值] 读取失败: {type(exc).__name__}: {exc}')

            prompts = [(task, '正确')]
            prompts += [(t, '其它') for t in all_tasks if t != task]
            prompts += [(WRONG_PROMPT, '旧/错')]

            ref = None
            gt_first = None
            for prompt, kind in prompts:
                o = dict(obs)
                o['task_description'] = prompt
                times_s, times_w = [], []
                a = None
                for _ in range(max(1, args.repeat)):
                    a, t_srv, t_wall = predict(
                        sock, serializer, o, unnorm_key=args.unnorm_key,
                        seed=args.seed)
                    times_s.append(t_srv)
                    times_w.append(t_wall)
                info = summarise(a)
                fl = '' if info['finite'] else '  ⚠NaN'
                if ref is None:
                    ref = a
                    diff_s = '  --'
                else:
                    diff_s = f'{float(np.abs(a - ref).max()):8.4f}'

                # 相对真值的右臂 MAE（P0 的量化指标）
                mae_val = None
                if gt_abs is not None:
                    n = min(a.shape[0], gt_abs.shape[0])
                    p16 = pred18_to_16(a[:n])
                    mae_val = float(np.abs(p16 - gt_abs[:n])[:, 8:15].mean())
                    if gt_first is None:
                        gt_first = (p16[0, 8:11], gt_abs[0, 8:11])
                mae_s = ('     n/a' if mae_val is None else f'{mae_val:8.4f}')

                tag = '' if kind == '正确' else f'  [{kind}]'
                print(f'  {prompt:38s} {np.median(times_s):9.1f} '
                      f'{np.median(times_w):8.1f} {info["r_arm"]:9.4f} '
                      f'{info["grip_l"]:8.4f} {info["grip_r"]:8.4f} '
                      f'{diff_s} {mae_s}{fl}{tag}')
                rows.append(dict(dataset=name, prompt=prompt, kind=kind,
                                 **{k2: v for k2, v in info.items()},
                                 diff_vs_correct=(
                                     None if kind == '正确'
                                     else float(np.abs(a - ref).max())),
                                 mae_vs_gt=mae_val,
                                 srv_ms=float(np.median(times_s)),
                                 wall_ms=float(np.median(times_w))))

            if gt_first is not None:
                print(f'  [真值第0步] 预测R7[0:3]='
                      f'{np.round(gt_first[0], 4).tolist()}  真值='
                      f'{np.round(gt_first[1], 4).tolist()}')

        # ---------- 4. 汇总 ----------
        print('\n' + '=' * 78)
        print('汇总')
        print('=' * 78)
        shapes = {r['shape'] for r in rows}
        print(f'\n输出形状        : {shapes}   (期望 (50, 18))')
        print(f'全部 finite     : {all(r["finite"] for r in rows)}')
        print(f'左夹爪恒为 0    : {all(abs(r["grip_l"]) < 1e-4 for r in rows)}')
        print(f'服务端延迟中位  : {np.median([r["srv_ms"] for r in rows]):.1f} ms')
        print(f'往返延迟中位    : {np.median([r["wall_ms"] for r in rows]):.1f} ms')

        for kind, label in (('正确', '正确 prompt'), ('其它', '换其它任务'),
                            ('旧/错', '旧/错 prompt')):
            sub = [r for r in rows if r['kind'] == kind]
            if not sub:
                continue
            ms = [r['mae_vs_gt'] for r in sub if r['mae_vs_gt'] is not None]
            ds = [r['diff_vs_correct'] for r in sub
                  if r['diff_vs_correct'] is not None]
            line = f'{label:14s} 右臂 MAE vs 真值 中位 = ' + (
                f'{np.median(ms):.4f} rad' if ms else 'n/a')
            if ds:
                line += f'   (动作偏移中位 {np.median(ds):.4f})'
            print(line)

        print('\n结论：')
        print('  - 返回的 18 维是【绝对关节角】—— delta 已被 '
              'DenormalizeDeltaAction 还原，')
        print('    与数据集原始 action 同量纲，可直接比 MAE')
        print('  - 换 prompt 后「vs正确」> 0，说明 prompt 在按任务路由')
    finally:
        sock.close()
        ctx.term()


if __name__ == '__main__':
    main()
