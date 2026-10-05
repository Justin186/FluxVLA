#!/usr/bin/env python
# =============================================================================
#  TRON2 三按钮数据集 —— 用末端笛卡尔位姿定位「每个任务到底按在哪」
#
#  为什么需要：
#    实机表现是「让按 green 却一直在按 black」。这有两种可能：
#      (a) 模型不会用 prompt 路由（训练不够 / prompt 信号太弱）
#      (b) 数据集标注本身有问题 —— 标着 green 的录制内容其实在按 black
#    只要把三个数据集的【按压点】在笛卡尔空间里画出来就能区分。
#
#  做法：
#    数据集带 observation.ee_pose_right（7 维 = xyz + 四元数）。
#    对每集取「伸得最远的那一帧」当作按压时刻，收集右臂末端 xyz，
#    再比较三个数据集的均值和离散度。
#
#  判读：
#    - 三个簇清晰分开        → 标注没问题，是模型路由问题 (a)
#    - 某两簇重合            → 标注有问题 (b)，那两任务的录制内容其实一样
#    - 簇间距 < 簇内离散度   → 任务在物理上不可分，模型只能猜
#
#  用法：
#    python scripts/diagnose_tron2_button_location.py
# =============================================================================
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# 仓库根目录：默认取本脚本所在目录的上一级，任何机器都能跑；可用 FLUXVLA_ROOT 覆盖
FV = os.environ.get(
    'FLUXVLA_ROOT',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(FV, 'datasets/RealRobot_Tron2_lerobot')


def press_position(ee):
    """ee: (T, 7) 右臂末端位姿 -> 按压帧的 xyz 与「离起始最远」的距离。"""
    pos = ee[:, 0:3]
    dist = np.linalg.norm(pos - pos[0], axis=1)
    k = int(np.argmax(dist))
    return pos[k], dist[k], k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=120,
                    help='每个数据集抽样多少集')
    ap.add_argument('--root', default=ROOT)
    args = ap.parse_args()

    result = {}
    for d in sorted(glob.glob(os.path.join(args.root, 'lerobot_*'))):
        name = os.path.basename(d)
        task = pq.read_table(os.path.join(d, 'meta', 'tasks.parquet')
                             ).to_pandas()
        task_text = task['task'].tolist()
        # task_index 映射检查
        ti = task['index'].tolist() if 'index' in task.columns else None
        files = sorted(glob.glob(os.path.join(d, 'data', '**', '*.parquet'),
                                 recursive=True))[:args.sample]
        pts, reach, idx_used = [], [], set()
        for f in files:
            try:
                t = pq.read_table(
                    f, columns=['observation.ee_pose_right', 'task_index']
                ).to_pandas()
            except Exception:                                   # noqa: BLE001
                continue
            ee = np.stack([np.asarray(x, np.float64)
                           for x in t['observation.ee_pose_right']])
            if len(ee) < 20:
                continue
            p, r, _ = press_position(ee)
            pts.append(p)
            reach.append(r)
            idx_used.update(np.unique(t['task_index'].to_numpy()).tolist())
        result[name] = dict(task=task_text, pts=np.asarray(pts),
                            reach=np.asarray(reach), task_index=ti,
                            idx_used=sorted(idx_used))

    print('=' * 78)
    print('一、task_index 映射检查（数据里的 index -> tasks.parquet 的文本）')
    print('=' * 78)
    for name, r in result.items():
        print(f'  {name}')
        print(f'     tasks.parquet : index={r["task_index"]} task={r["task"]}')
        print(f'     数据里的 task_index 取值: {r["idx_used"]}')

    print()
    print('=' * 78)
    print('二、按压点（右臂末端 xyz，单位 m）—— 「伸得最远的那一帧」')
    print('=' * 78)
    print(f'  {"数据集":30s} {"n":>4s} {"x":>18s} {"y":>18s} {"z":>18s} '
          f'{"伸展":>8s}')
    for name, r in result.items():
        P = r['pts']
        if not len(P):
            continue
        m, s = P.mean(0), P.std(0)
        print(f'  {os.path.basename(name):30s} {len(P):4d} '
              f'{m[0]:+8.4f}±{s[0]:.4f} {m[1]:+8.4f}±{s[1]:.4f} '
              f'{m[2]:+8.4f}±{s[2]:.4f} {r["reach"].mean():8.4f}')
        r['mean'], r['std'] = m, s

    print()
    print('=' * 78)
    print('三、两两可分性（簇间距 vs 簇内离散度）')
    print('=' * 78)
    print('  可分性 = 中心距 / 合并标准差。> 2 才算「清晰可分」，'
          '< 1.5 基本是同一片区域。')
    print()
    keys = list(result)
    for a, b in itertools.combinations(keys, 2):
        ra, rb = result[a], result[b]
        if 'mean' not in ra or 'mean' not in rb:
            continue
        d = float(np.linalg.norm(ra['mean'] - rb['mean']))
        pooled = float(np.sqrt((ra['std'] ** 2 + rb['std'] ** 2).mean()))
        ratio = d / max(pooled, 1e-9)
        flag = ('  ← 几乎重合！' if ratio < 1.5 else
                ('  ← 可分' if ratio > 2 else '  ← 勉强'))
        print(f'  {ra["task"][0][:26]:28s} vs {rb["task"][0][:26]:28s} '
              f'距={d:.4f} m  合并std={pooled:.4f}  可分性={ratio:.2f}{flag}')

    print()
    print('=' * 78)
    print('四、把三个数据集的按压点混在一起，看能分成几簇')
    print('=' * 78)
    allpts = np.vstack([r['pts'] for r in result.values() if len(r['pts'])])
    labels = np.concatenate([[os.path.basename(n)] * len(r['pts'])
                             for n, r in result.items() if len(r['pts'])])
    # 简单最近中心判定：用「真实标签」的中心算分类准确率
    means = np.array([r['mean'] for r in result.values()
                      if 'mean' in r])
    names = [n for n, r in result.items() if 'mean' in r]
    D = np.linalg.norm(allpts[:, None, :] - means[None, :, :], axis=2)
    pred = D.argmin(1)
    gt = np.array([names.index(l) for l in labels])
    # 混淆矩阵
    hdr = '真实 / 预测'
    print(f'  {hdr:30s} ' +
          ' '.join(f'{os.path.basename(n)[-8:]:>10s}' for n in names))
    for i, n in enumerate(names):
        cnt = [int(((gt == i) & (pred == j)).sum()) for j in range(len(names))]
        tot = max(sum(cnt), 1)
        print(f'  {os.path.basename(n):30s} ' +
              ' '.join(f'{c:5d}({100 * c / tot:3.0f}%)' for c in cnt))
    print()
    print(f'  用「每个数据集自己的均值」做最近邻分类的准确率: '
          f'{100 * (gt == pred).mean():.1f}%')
    print('  如果某个数据集的点大量被判给另一个数据集，说明这两簇重合。')


if __name__ == '__main__':
    main()
