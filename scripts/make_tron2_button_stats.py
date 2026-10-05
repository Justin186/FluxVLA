#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重算 TRON2 归一化统计量，并施加「退化维」修正（用于三个按钮任务的训练）。

背景（详见 tron2_cabinet_lora_rebuild_guide.md §7.2）：
  * 左臂 7 维在采集里全程闲置，其 relative action 的 q01-q99 跨度塌缩到 ~0.005，
    quantile 归一化 (x-q01)/(q99-q01+eps)*2-1 把噪声放大约 150~280 倍，
    制造出与任务无关的 loss 地板和尖峰。
  * 左夹爪恒为 0 -> q01==q99==0，归一化分母只剩 1e-6 兜底；当前恰好无害，
    但只要读数有一丁点抖动就会被放大到满量程。
  * 只训练三个按钮（去掉拧旋钮）后，右夹爪也从 span~0.94 退化成 {0, 0.02}
    两个值（span=0.02），同样被铺满成 ±1 的二值噪声目标。

本脚本做两处修正（都只改 q01/q99，不改 mean/std/min/max）：
  1. 左臂 0-6 维的 q01/q99 <- 右臂对称关节 8-14 维（同名关节运动范围相同）。
  2. 两个夹爪维（7 和 15）的 q01/q99 -> 物理量程 [0, 1]（数据集里夹爪就是 0-1）。
     -> 右夹爪的归一化变化从 ±1 压到 ~0.04，对 loss 几乎无贡献。

用法：
    python3 make_tron2_button_stats.py \
        --fluxvla ../FluxVLA \
        --data-dir ../FluxVLA/datasets/RealRobot_Tron2_lerobot \
        --out ../FluxVLA/datasets/RealRobot_Tron2_lerobot/tron2_stats_armsymmetric.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DIM_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'gripL',
             'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'gripR']
ARM_L = list(range(0, 7))       # 左臂
GRIP_L = 7                      # 左夹爪
ARM_R = list(range(8, 15))      # 右臂
GRIP_R = 15                     # 右夹爪


def discover(data_dir, subsets):
    if subsets:
        names = [s.strip() for s in subsets.split(',') if s.strip()]
    else:
        names = sorted(d for d in os.listdir(data_dir)
                       if os.path.isfile(os.path.join(data_dir, d, 'meta', 'info.json')))
    paths = []
    for n in names:
        p = os.path.join(data_dir, n)
        if not os.path.isfile(os.path.join(p, 'meta', 'info.json')):
            raise SystemExit('子集不存在或缺少 meta/info.json: %s' % p)
        paths.append(p)
    if not paths:
        raise SystemExit('在 %s 下没找到任何子集' % data_dir)
    return names, paths


def print_table(tag, q01, q99):
    print('  [%s] amp=1/span（越大越危险）' % tag)
    for i, n in enumerate(DIM_NAMES):
        span = float(q99[i] - q01[i])
        amp = ('inf' if span < 1e-9 else '%.1f' % (1.0 / span))
        print('    %-6s q01=%9.5f q99=%9.5f span=%9.5f amp=%s'
              % (n, q01[i], q99[i], span, amp))


def main():
    ap = argparse.ArgumentParser(description='重算并修正 TRON2 按钮任务统计量')
    ap.add_argument('--fluxvla', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'FluxVLA'),
        help='FluxVLA 仓库根目录（用于 import fluxvla）')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--subsets', default='',
                    help='逗号分隔的子集名；默认扫描 data-dir 下全部合法子集')
    ap.add_argument('--action-horizon', type=int, default=50)
    ap.add_argument('--window-start-index', type=int, default=0)
    ap.add_argument('--statistic-name', default='private')
    ap.add_argument('--gripper-range', default='0,1',
                    help='夹爪维钉死的量化区间，逗号分隔 LOW,HIGH；默认 0,1')
    ap.add_argument('--no-arm-symmetric', action='store_true',
                    help='关闭「左臂抄右臂」修正（默认开启）')
    ap.add_argument('--no-gripper-pin', action='store_true',
                    help='关闭「夹爪钉物理量程」修正（默认开启）')
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.fluxvla))
    from fluxvla.datasets.utils.transformed_statistics import compute_statistics

    names, paths = discover(args.data_dir, args.subsets)
    print('数据集 (%d 个):' % len(names))
    for n in names:
        print('   -', n)

    stats, meta = compute_statistics(
        [Path(p) for p in paths],
        profile_name='tron2',
        action_horizon=args.action_horizon,
        window_start_idx=args.window_start_index,
        repeat_terminal=True,
        statistic_name=args.statistic_name)
    print('\n自动统计完成: episodes=%d state_samples=%d action_samples=%d'
          % (meta['episodes'], meta['state_samples'], meta['action_samples']))

    lo, hi = (float(x) for x in args.gripper_range.split(','))
    fixes = {}
    grp = stats[args.statistic_name]
    for key in ('proprio', 'action'):
        s = grp[key]
        q01 = np.asarray(s['q01'], dtype=np.float64)
        q99 = np.asarray(s['q99'], dtype=np.float64)
        print('\n修正前:')
        print_table(key, q01, q99)

        if not args.no_arm_symmetric:
            q01[ARM_L] = q01[ARM_R]
            q99[ARM_L] = q99[ARM_R]
            fixes['arm_symmetric'] = 'left-arm dims 0-6 reuse right-arm dims 8-14 q01/q99'
        if not args.no_gripper_pin:
            for g in (GRIP_L, GRIP_R):
                q01[g] = lo
                q99[g] = hi
            fixes['gripper_pin'] = 'gripper dims 7/15 pinned to [%g, %g]' % (lo, hi)

        s['q01'] = q01.tolist()
        s['q99'] = q99.tolist()
        print('\n修正后:')
        print_table(key, q01, q99)

    grp['_fix_note'] = {
        'reason': ('Left arm is idle in the button datasets (relative-action '
                   'q01-q99 span ~0.005) and grippers carry no relevant signal '
                   'for button pressing; quantile normalization would amplify '
                   'their noise to full range.'),
        'fixes': fixes,
        'datasets': names,
        'action_horizon': args.action_horizon,
        'window_start_index': args.window_start_index,
        'repeat_terminal': True,
    }

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print('\n已写出: %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
