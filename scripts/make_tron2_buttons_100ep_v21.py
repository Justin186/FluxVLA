#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 TRON2 三个颜色按钮子集中各抽 N 个 episode，合成一个 LeRobot v2.1 数据集。

背景
----
源数据是数采软件导出的 **LeRobot v3.0**，且满足两个便利条件：
  * 每个 episode 对应 1 个 data parquet、每个相机对应 1 个 mp4
    （``file_index == episode_index``，已在 prepare_tron2_for_fluxvla.py 里验证）；
  * parquet 的 ``timestamp`` 已与视频时间轴对齐（前导黑帧被丢弃但未重编码视频，
    所以 timestamp 从 d/fps 开始，正好指向视频中的正确画面）。

因此 v3.0 -> v2.1 只差 **目录布局/文件命名** 与 **meta 清单格式**，
视频无需重编码，本脚本：

  1) 每个颜色子集随机抽 N 集（固定 seed，可复现）；
  2) 全局重编号 episode_index，并按颜色分配 task_index；
  3) 按 v2.1 布局搬移 parquet（重写 episode_index / index / task_index），
     视频用硬链接（同盘零拷贝，失败回退符号链接/复制）；
  4) 重写 meta：info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl。

用法
----
    python3 scripts/make_tron2_buttons_100ep_v21.py \
        --out FluxVLA/datasets/RealRobot_Tron2_buttons_100ep_v21

    # 换抽样集数 / 随机种子 / 数据源
    python3 scripts/make_tron2_buttons_100ep_v21.py \
        --per-subset 50 --seed 7 --out /tmp/tron2_50ep_v21
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DEFAULT_FLUXVLA = HERE.parent.resolve()
DEFAULT_DATA_DIR = 'datasets/RealRobot_Tron2_lerobot'

CHUNKS_SIZE = 1000
V21_DATA_PATH = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
V21_VIDEO_PATH = ('videos/chunk-{episode_chunk:03d}/{video_key}/'
                  'episode_{episode_index:06d}.mp4')

# 源 v3 parquet 里由本脚本重写的标量列
REWRITE_COLS = ('episode_index', 'index', 'task_index')


def to_jsonable(obj):
    """把 numpy / pyarrow 标量递归转成 JSON 可序列化对象。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, (np.bool_, )):
        return bool(obj)
    return obj


def discover_subsets(data_dir):
    subs = sorted(d for d in os.listdir(data_dir)
                  if os.path.isfile(os.path.join(data_dir, d, 'meta', 'info.json')))
    if not subs:
        raise SystemExit('在 %s 下没找到任何 LeRobot 子集' % data_dir)
    return subs


def load_info(subset_dir):
    with open(subset_dir / 'meta' / 'info.json', encoding='utf-8') as f:
        return json.load(f)


def load_episodes_meta(subset_dir):
    files = sorted((subset_dir / 'meta' / 'episodes').rglob('*.parquet'))
    if not files:
        raise SystemExit('找不到 %s/meta/episodes/*.parquet' % subset_dir)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values('episode_index').reset_index(drop=True)


def load_task_text(subset_dir):
    tp = subset_dir / 'meta' / 'tasks.parquet'
    if tp.is_file():
        tdf = pd.read_parquet(tp)
        for col in ('task', 'tasks', '__index_level_0__'):
            if col in tdf.columns:
                val = tdf[col].iloc[0]
                if val is not None:
                    return ' '.join(str(val).split()).strip()
        return ' '.join(str(tdf.index[0]).split()).strip()
    return ''


def link_or_copy(src, dst, mode):
    """把 src 放到 dst：hardlink(默认)/symlink/copy，硬链接失败自动回退。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'copy':
        shutil.copy2(src, dst)
        return 'copy'
    if mode == 'symlink':
        os.symlink(os.path.abspath(src), dst)
        return 'symlink'
    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        os.symlink(os.path.abspath(src), dst)
        return 'symlink'


def video_keys_of(info):
    return [k for k in info.get('features', {})
            if k.startswith('observation.images.')]


def rewrite_frame_table(tbl, new_ep, task_idx, start_index):
    """重写 episode_index / index / task_index，返回 (表, 帧数, 下一个全局 index)。"""
    n = tbl.num_rows
    cols = list(tbl.column_names)
    values = {
        'episode_index': pa.array(np.full(n, new_ep, dtype=np.int64)),
        'index': pa.array(np.arange(start_index, start_index + n, dtype=np.int64)),
        'task_index': pa.array(np.full(n, task_idx, dtype=np.int64)),
    }
    for col, arr in values.items():
        if col in cols:
            tbl = tbl.set_column(cols.index(col), col, arr)
        else:
            tbl = tbl.append_column(col, arr)
    return tbl, n, start_index + n


def to_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(
        description='从 TRON2 三个颜色按钮子集各抽 N 集，合成 LeRobot v2.1 数据集')
    ap.add_argument('--fluxvla', default=str(DEFAULT_FLUXVLA),
                    help='FluxVLA 仓库根目录')
    ap.add_argument('--data-dir', default=None,
                    help='源数据目录（默认 <fluxvla>/datasets/RealRobot_Tron2_lerobot）')
    ap.add_argument('--out', required=True, help='输出数据集目录')
    ap.add_argument('--per-subset', type=int, default=100,
                    help='每个颜色子集抽取的 episode 数（默认 100）')
    ap.add_argument('--seed', type=int, default=42, help='抽样随机种子')
    ap.add_argument('--link', choices=['hardlink', 'symlink', 'copy'],
                    default='hardlink', help='视频放置方式（默认硬链接，不占额外空间）')
    ap.add_argument('--dry-run', action='store_true', help='只打印计划，不写文件')
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else (
        Path(args.fluxvla) / DEFAULT_DATA_DIR)
    out = Path(args.out).resolve()
    if not data_dir.is_dir():
        raise SystemExit('源数据目录不存在: %s' % data_dir)

    subsets = discover_subsets(data_dir)
    print('源数据目录 : %s' % data_dir)
    print('发现子集   : %s' % subsets)
    print('输出目录   : %s' % out)
    print('每集抽取   : %d (seed=%d)' % (args.per_subset, args.seed))

    # ---------- 1) 抽样 ----------
    plan = []          # (subset_name, episode_index, sub, info)
    excluded = []
    meta_cache = {}
    for si, name in enumerate(subsets):
        sub = data_dir / name
        meta = load_episodes_meta(sub)
        meta_cache[name] = meta
        info = load_info(sub)
        # 只保留标记为 success 的 episode（没有该列则全收）
        if 'episode_success' in meta.columns:
            ok_mask = meta['episode_success'].astype(str).str.lower().isin(
                ('success', 'true', '1'))
            bad = int((~ok_mask).sum())
            if bad:
                excluded.append('%s: 跳过 %d 个非 success episode' % (name, bad))
            meta = meta[ok_mask].reset_index(drop=True)
        eps = meta['episode_index'].astype(int).tolist()
        k = min(args.per_subset, len(eps))
        if k < args.per_subset:
            print('  [WARN] %s 只有 %d 个 episode，少于请求的 %d'
                  % (name, len(eps), args.per_subset))
        rng = random.Random(args.seed + si)
        picked = sorted(rng.sample(eps, k))
        for ep in picked:
            plan.append((name, ep, sub, info))

    print('将写入 %d 个 episode' % len(plan))
    for msg in excluded:
        print('  [INFO] %s' % msg)
    for i, (name, ep, _, _) in enumerate(plan[:5]):
        print('  e.g. new_ep=%d <- %s/ep%03d' % (i, name, ep))
    if len(plan) > 5:
        print('  ...')

    if args.dry_run:
        print('\n[dry-run] 未写入任何文件')
        return 0

    # ---------- 2) 准备输出目录 ----------
    if out.exists():
        print('输出目录已存在，先删除: %s' % out)
        shutil.rmtree(out)
    (out / 'meta').mkdir(parents=True)
    (out / 'data').mkdir(parents=True)
    (out / 'videos').mkdir(parents=True)

    # ---------- 3) 逐 episode 落盘 ----------
    # task 文本 -> task_index（按子集出现顺序编号，稳定可复现）
    task_texts = []
    for name in subsets:
        text = load_task_text(data_dir / name)
        if text not in task_texts:
            task_texts.append(text)
    task_index_of = {t: i for i, t in enumerate(task_texts)}
    task_index_of_subset = {
        name: task_index_of[load_task_text(data_dir / name)] for name in subsets
    }
    print('任务标签 (%d):' % len(task_texts))
    for i, t in enumerate(task_texts):
        print('  [%d] %r' % (i, t))

    link_mode_used = {}
    episodes_jsonl = []
    episodes_stats_jsonl = []
    global_index = 0
    total_frames = 0
    n_videos = 0

    for new_ep, (name, ep, sub, info) in enumerate(plan):
        meta = meta_cache[name]
        row = meta[meta['episode_index'].astype(int) == ep].iloc[0]
        chunk = new_ep // CHUNKS_SIZE
        task_idx = task_index_of_subset[name]

        # --- data parquet ---
        src_rel = info['data_path'].format(
            chunk_index=int(row.get('data/chunk_index', 0)),
            file_index=int(row.get('data/file_index', ep)),
            episode_chunk=chunk, episode_index=ep)
        src_parquet = sub / src_rel
        if not src_parquet.is_file():
            raise SystemExit('找不到源 parquet: %s' % src_parquet)
        tbl = pq.read_table(src_parquet)
        tbl, n_frames, global_index = rewrite_frame_table(
            tbl, new_ep, task_idx, global_index)
        dst_parquet = out / V21_DATA_PATH.format(
            episode_chunk=chunk, episode_index=new_ep)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, dst_parquet, compression='snappy')
        total_frames += n_frames

        # --- videos ---
        for vk in video_keys_of(info):
            src_vid = sub / info['video_path'].format(
                video_key=vk,
                episode_chunk=int(row.get('videos/%s/chunk_index' % vk, 0)),
                chunk_index=int(row.get('data/chunk_index', 0)),
                episode_index=int(row.get('videos/%s/file_index' % vk, ep)),
                file_index=int(row.get('videos/%s/file_index' % vk, ep)))
            if not src_vid.is_file():
                raise SystemExit('找不到源视频: %s' % src_vid)
            dst_vid = out / V21_VIDEO_PATH.format(
                episode_chunk=chunk, video_key=vk, episode_index=new_ep)
            used = link_or_copy(src_vid, dst_vid, args.link)
            link_mode_used[used] = link_mode_used.get(used, 0) + 1
            n_videos += 1

        # --- episode 清单 ---
        length = int(row.get('length', n_frames))
        episodes_jsonl.append({
            'episode_index': new_ep,
            'tasks': [task_texts[task_idx]],
            'length': length,
        })

        # --- episode 统计 ---
        stats = {}
        for col in row.index:
            if col.startswith('stats/'):
                stats[col.split('/', 1)[1]] = to_jsonable(row[col])
        episodes_stats_jsonl.append({
            'episode_index': new_ep,
            'stats': stats,
        })

    # ---------- 4) meta ----------
    src_info = load_info(data_dir / subsets[0])
    features = {k: to_jsonable(v) for k, v in src_info['features'].items()}
    info_out = {
        'codebase_version': 'v2.1',
        'robot_type': src_info.get('robot_type', 'tron2'),
        'total_episodes': len(plan),
        'total_frames': total_frames,
        'total_tasks': len(task_texts),
        'total_videos': n_videos,
        'total_chunks': (len(plan) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        'chunks_size': CHUNKS_SIZE,
        'fps': src_info.get('fps', 30),
        'splits': {'train': '0:%d' % len(plan)},
        'data_path': V21_DATA_PATH,
        'video_path': V21_VIDEO_PATH,
        'features': features,
    }
    with open(out / 'meta' / 'info.json', 'w', encoding='utf-8') as f:
        json.dump(info_out, f, ensure_ascii=False, indent=4)

    to_jsonl(out / 'meta' / 'tasks.jsonl',
             [{'task_index': i, 'task': t} for i, t in enumerate(task_texts)])
    to_jsonl(out / 'meta' / 'episodes.jsonl', episodes_jsonl)
    to_jsonl(out / 'meta' / 'episodes_stats.jsonl', episodes_stats_jsonl)

    # 源 v3 的 stats.json（非 v2.1 标准）如果存在，一并转过去方便对齐统计量
    src_stats = data_dir / subsets[0] / 'meta' / 'stats.json'
    if src_stats.is_file():
        with open(src_stats, encoding='utf-8') as f:
            with open(out / 'meta' / 'stats.json', 'w', encoding='utf-8') as g:
                json.dump(to_jsonable(json.load(f)), g,
                          ensure_ascii=False, indent=4)

    # ---------- 5) 摘要 ----------
    print()
    print('=' * 70)
    print('完成: %d 个 episode, %d 帧, %d 个视频 (%s)'
          % (len(plan), total_frames, n_videos,
             ', '.join('%s=%d' % kv for kv in sorted(link_mode_used.items()))))
    print('输出: %s' % out)
    print('=' * 70)
    print('校验:  python3 scripts/check_lerobot_dataset.py %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
