#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pod 入口：把平台的 `gm-run`（= python）转成 torchrun 启动训练。

为什么需要这个文件
------------------
平台的启动指令是 `gm-run <脚本> <参数>`，而 `gm-run` 等价于 `python`。
但 FluxVLA **不能用 python 直接启动**：

    AttributeError: 'PureOverwatch' object has no attribute 'local_rank'
    （fluxvla/engines/runners/fsdp_train_runner.py）

runner 里调用 `overwatch.local_rank()`，而纯 python 启动得到的是
`PureOverwatch`，没有这个方法。必须先由 torchrun 建立分布式上下文。
所以这里不再自己跑训练，而是把进程**替换成** torchrun。

顺带处理两件 pod 上必须做、而平台默认启动不会做的事：

  * **丢掉 PYTHONPATH**：pod 镜像里有个 IDE 的 `sitecustomize.py` 钩住了
    `os.remove`，把它重定向到回收站。训练滚动删除旧检查点时空间不会释放，
    每存一次泄漏约 29 GB，**约 6 次后磁盘就满**。子进程环境里直接不带
    `PYTHONPATH`（比 unset 更彻底）。
  * **WANDB_MODE=disabled**：pod 上没有 `WANDB_API_KEY`，`wandb.init()` 会挂起。

用法（在 pod 上，仓库根目录）
----------------------------
    gm-run scripts/pod_train.py                       # 默认单卡 + 默认 work-dir
    gm-run scripts/pod_train.py --nproc-per-node 2    # 多卡
    gm-run scripts/pod_train.py --work-dir /personal/tron2_work
    gm-run scripts/pod_train.py --resume-from <step-XXXXXX.pt>
    gm-run scripts/pod_train.py --smoke               # 40 步冒烟，写到 /tmp/smoke

脚本名之后的所有参数都会原样转给 `scripts/train.py`，只有下面两个参数
由本脚本自己消费：

    --nproc-per-node N   用几张卡（默认 1，与平台"1×4090D"一致）
    --smoke              冒烟模式：40 步、写到 /tmp/smoke、不碰正式 work-dir

`--config` 与 `--work-dir` 由本脚本补齐，不需要在启动指令里写。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = 'configs/pi05/pi05_paligemma_tron2_cabinet_lora.py'
DEFAULT_WORK = 'work_dirs/tron2_buttons_v2'
SMOKE_WORK = '/tmp/smoke'
SMOKE_STEPS = 40


def find_torchrun() -> str:
    """Locate torchrun, preferring the interpreter's own environment.

    On the pod the interpreter is the conda env's python, so its sibling
    `torchrun` is the right one; PATH comes second because the platform may
    have injected its own.
    """
    candidates = [
        os.environ.get('FLUXVLA_TORCHRUN'),
        os.path.join(os.path.dirname(sys.executable), 'torchrun'),
        shutil.which('torchrun'),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    sys.exit('[FAIL] 找不到 torchrun。已试过：\n  ' +
             '\n  '.join(str(c) for c in candidates) +
             '\n请设置 FLUXVLA_TORCHRUN 指向它（例如 '
             '/opt/miniconda3/envs/fluxvla/bin/torchrun）')


def main() -> int:
    ap = argparse.ArgumentParser(
        add_help=False,
        description='pod 入口：torchrun 启动 TRON2 训练')
    ap.add_argument('--nproc-per-node', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help='只打印将要执行的命令并退出（在 pod 上先验证再正式跑）')
    ap.add_argument('-h', '--help', action='store_true')
    known, passthrough = ap.parse_known_args()

    if known.help:
        print(__doc__)
        return 0

    config = known.config
    if known.smoke:
        work_dir = known.work_dir or SMOKE_WORK
        passthrough += ['--cfg-options', 'runner.max_steps=%d' % SMOKE_STEPS]
    else:
        work_dir = known.work_dir or DEFAULT_WORK
    work_dir = os.path.abspath(work_dir)

    torchrun = find_torchrun()

    cmd = [torchrun,
           '--standalone', '--nnodes', '1',
           '--nproc-per-node', str(known.nproc_per_node),
           os.path.join(REPO, 'scripts/train.py'),
           '--config', config,
           '--work-dir', work_dir] + passthrough

    env = dict(os.environ)
    # 不带 PYTHONPATH -> IDE 的 os.remove shim 不会加载（否则检查点删不掉，
    # 每次保存泄漏 29 GB，约 6 次后磁盘满）
    env.pop('PYTHONPATH', None)
    env.setdefault('WANDB_MODE', 'disabled')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    print('=' * 78)
    print('  pod_train: 用 torchrun 启动（python 直接跑会崩在 local_rank）')
    print('=' * 78)
    print('  torchrun     : %s' % torchrun)
    print('  nproc        : %d' % known.nproc_per_node)
    print('  config       : %s' % config)
    print('  work_dir     : %s' % work_dir)
    print('  PYTHONPATH   : %s' % ('已丢弃 ✓' if 'PYTHONPATH' not in env
                                   else '!! 仍在，危险'))
    print('  WANDB_MODE   : %s' % env['WANDB_MODE'])
    if passthrough:
        print('  额外参数     : %s' % ' '.join(passthrough))
    print('-' * 78)
    print('  %s' % ' '.join(cmd))
    print('=' * 78)
    sys.stdout.flush()

    if known.dry_run:
        print('  [dry-run] 未启动、未创建任何目录。去掉 --dry-run 即正式运行。')
        return 0

    try:
        os.makedirs(work_dir, exist_ok=True)
    except OSError as exc:                                  # noqa: BLE001
        sys.exit('[FAIL] 无法创建 work_dir %s: %s\n'
                 '       pod 上如果想让检查点持久化，用 --work-dir 指向已挂载的 '
                 '/personal/ 下的目录。' % (work_dir, exc))

    # execvpe: 用 torchrun 替换当前进程，保持同一个 PID，
    # 这样平台的进程监控 / 停止按钮作用在真正的训练进程上。
    # 在 pod 上绝对不要用 sudo。
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as exc:                                  # noqa: BLE001
        sys.exit('[FAIL] exec %s 失败: %s' % (cmd[0], exc))
    return 0                                                   # pragma: no cover


if __name__ == '__main__':
    raise SystemExit(main())
