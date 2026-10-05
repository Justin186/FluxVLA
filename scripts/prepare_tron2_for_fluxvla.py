#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把数采软件导出的 LeRobot v3.0 数据（TRON2，21 维）转成 FluxVLA TRON2 配置可直接吃掉的形态。

本脚本做 5 件事：
  1. observation.state / action 从 21 维裁到前 16 维
     [左臂7, 左夹爪, 右臂7, 右夹爪]  ← 正好是 FluxVLA 的 ori_action_dim=16
     丢弃 [16:21] = 头部2 + 底盘2 + 升降1
  2. info.json 的 video_path 模板改成 ProcessParquetInputs 能填的占位符
     原: videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4   → KeyError
     改: videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4
  3. tasks.parquet 重写成标准 'task' 列，并清洗零宽空格 (U+200B 等)
  4. stats.json / meta/episodes 里的 stats/* 一并裁到 16 维，保持一致
  5. 顺手修掉每段起始的"全零无效帧"（关节状态还没上报就被采样）

视频文件用符号链接，不复制、不重编码。
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEEP_DIM = 16

# 各类不可见字符：零宽空格、BOM、方向标记等
INVISIBLE = ("\u200b", "\u200c", "\u200d", "\ufeff",
             "\u200e", "\u200f", "\u202a", "\u202b",
             "\u202c", "\u202d", "\u202e")


def clean_text(s):
    s = str(s)
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    return " ".join(s.split()).strip()


def trim_stats_dict(d, n=KEEP_DIM):
    """把 stats 字典里的 per-dim 数组裁到前 n 维"""
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, np.ndarray)):
            out[k] = np.asarray(v, dtype=np.float64).ravel()[:n].tolist()
        else:
            out[k] = v
    return out


def read_episodes(meta_dir):
    """读 meta/episodes/**/*.parquet，返回 (DataFrame, 文件列表)"""
    import pandas as pd
    base = os.path.join(meta_dir, "episodes")
    files = []
    for dirpath, _, filenames in os.walk(base):
        for fn in filenames:
            if fn.endswith(".parquet"):
                files.append(os.path.join(dirpath, fn))
    if not files:
        return None, []
    frames = [pd.read_parquet(f) for f in sorted(files)]
    return pd.concat(frames, ignore_index=True), sorted(files)


def check_file_index_consistency(ep_df):
    """验证 file_index == episode_index（改用 episode_index 拼视频路径的前提）"""
    need = {"episode_index", "videos/observation.images.cam_high/file_index"}
    if not need.issubset(set(ep_df.columns)):
        return None, "meta/episodes 里找不到 file_index 列，无法验证"
    bad = []
    for _, row in ep_df.iterrows():
        if int(row["videos/observation.images.cam_high/file_index"]) != int(row["episode_index"]):
            bad.append(int(row["episode_index"]))
    if bad:
        return False, "有 %d 个 episode 的 file_index != episode_index（如 %s）" % (
            len(bad), bad[:5])
    return True, "file_index == episode_index 全部成立"


def fix_zero_frames(mat, ep_ids):
    """把每段起始处"16 维全零"的无效帧替换成同段下一帧的值"""
    fixed = 0
    n = mat.shape[0]
    for i in range(n):
        if np.abs(mat[i]).sum() == 0.0:
            # 找同段下一个有效帧
            j = i + 1
            while j < n and ep_ids[j] == ep_ids[i] and np.abs(mat[j]).sum() == 0.0:
                j += 1
            if j < n and ep_ids[j] == ep_ids[i]:
                mat[i] = mat[j]
                fixed += 1
            else:
                # 找不到后继，用同段前一帧
                k = i - 1
                while k >= 0 and ep_ids[k] == ep_ids[i] and np.abs(mat[k]).sum() == 0.0:
                    k -= 1
                if k >= 0 and ep_ids[k] == ep_ids[i]:
                    mat[i] = mat[k]
                    fixed += 1
    return mat, fixed


def scan_black_runs(src, cams, max_check=12):
    """逐段扫描视频开头，返回 {episode_index: 需要丢弃的前导帧数}

    腕部相机在 episode 启动时有 1~7 帧延迟，期间输出纯黑图 (mean ≈ 0)；
    头部相机没有这个问题。取该段所有相机里最长的前导黑帧数作为丢弃量，
    这样绝大多数段丢 0 帧，只有真正黑帧的段才丢。
    """
    try:
        import av
    except ImportError:
        print("  [WARN] 没装 av，无法扫描黑帧")
        return None
    need = {}
    for cam in cams:
        vids = sorted(glob.glob(os.path.join(src, "videos", cam,
                                             "chunk-*", "*.mp4")))
        for vf in vids:
            fn = os.path.basename(vf)
            try:
                ep = int(fn.replace("file-", "").replace(".mp4", ""))
            except ValueError:
                continue
            run = 0
            try:
                c = av.open(vf)
                for i, fr in enumerate(c.decode(video=0)):
                    if i >= max_check:
                        break
                    if float(fr.to_ndarray(format="rgb24").mean()) < 1.0:
                        run = i + 1
                    else:
                        break
                c.close()
            except Exception:
                continue
            need[ep] = max(need.get(ep, 0), run)
    return need


def process_dataset(src, dst, drop_first=3):
    import pandas as pd

    print("=" * 74)
    print("源: %s" % src)
    print("目标: %s" % dst)

    if os.path.exists(dst):
        print("  目标已存在，先删除")
        shutil.rmtree(dst)
    os.makedirs(dst)

    meta_src = os.path.join(src, "meta")
    meta_dst = os.path.join(dst, "meta")
    os.makedirs(meta_dst)

    # ---------- 读 info ----------
    with open(os.path.join(meta_src, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    print("  源 info: version=%s robot=%s episodes=%s frames=%s"
          % (info.get("codebase_version"), info.get("robot_type"),
             info.get("total_episodes"), info.get("total_frames")))

    # ---------- 读 episodes ----------
    ep_df, ep_files = read_episodes(meta_src)
    if ep_df is None:
        print("  [FAIL] 找不到 meta/episodes")
        return False
    ok, msg = check_file_index_consistency(ep_df)
    print("  file_index 一致性: %s" % msg)
    if ok is False:
        print("  [FAIL] 不能安全改用 episode_index 拼视频路径，需要改用软链接方案")
        return False

    # ---------- 处理数据 parquet ----------
    data_src = os.path.join(src, "data")
    data_dst = os.path.join(dst, "data")
    n_files = 0
    n_zero_fixed = 0
    n_frames = 0
    n_dropped = 0
    src_dims = {}

    # 决定每段要丢多少前导帧
    ep_drop = {}
    if isinstance(drop_first, int):
        drop_desc = "固定丢每段前 %d 帧" % drop_first
    else:
        print("  扫描视频，逐段确定前导黑帧数 ...")
        ep_drop = scan_black_runs(
            src, [k for k in info["features"]
                  if k.startswith("observation.images.")]) or {}
        pos = [v for v in ep_drop.values() if v > 0]
        drop_desc = "自适应 (需丢帧的段 %d/%d, 最多 %d 帧)" % (
            len(pos), len(ep_drop), max(ep_drop.values()) if ep_drop else 0)
    print("  %s" % drop_desc)

    for dirpath, dirnames, filenames in os.walk(data_src):
        rel = os.path.relpath(dirpath, data_src)
        out_dir = os.path.join(data_dst, rel) if rel != "." else data_dst
        os.makedirs(out_dir, exist_ok=True)
        for fn in sorted(filenames):
            if not fn.endswith(".parquet"):
                continue
            src_f = os.path.join(dirpath, fn)
            dst_f = os.path.join(out_dir, fn)
            tbl = pq.read_table(src_f)

            # 丢掉每段开头"相机还没出图"的帧（腕部相机有 1~7 帧启动延迟 -> 纯黑）
            # 同时覆盖"第 0 帧关节状态全零"。视频文件不动，只裁 parquet 行：
            # 时间戳对齐靠 timestamp 列，被丢掉的 timestamp 之后不会再被请求。
            if "frame_index" in tbl.column_names:
                fi = np.asarray(tbl["frame_index"].to_pylist(), dtype=np.int64)
                if isinstance(drop_first, int):
                    need = np.full(len(fi), max(drop_first, 0), dtype=np.int64)
                else:
                    ei = np.asarray(tbl["episode_index"].to_pylist(), dtype=np.int64)
                    need = np.array([ep_drop.get(int(e), 0) for e in ei],
                                    dtype=np.int64)
                keep = fi >= need
                nd = int((~keep).sum())
                if nd:
                    tbl = tbl.filter(pa.array(keep))
                    if "index" in tbl.column_names:
                        tbl = tbl.set_column(
                            tbl.column_names.index("index"), "index",
                            pa.array(np.arange(tbl.num_rows, dtype=np.int64)))
                    n_dropped += nd

            ep_ids = (np.asarray(tbl["episode_index"].to_pylist())
                      if "episode_index" in tbl.column_names
                      else np.zeros(tbl.num_rows, dtype=np.int64))

            for col in ("observation.state", "action"):
                if col not in tbl.column_names:
                    continue
                arrs = [np.asarray(x.as_py(), dtype=np.float32).ravel()
                        for x in tbl[col]]
                src_dims[col] = len(arrs[0]) if arrs else 0
                mat = np.vstack(arrs)
                if mat.shape[1] > KEEP_DIM:
                    mat = mat[:, :KEEP_DIM]
                if col == "observation.state":
                    mat, fixed = fix_zero_frames(mat, ep_ids)
                    n_zero_fixed += fixed
                lst = pa.array([row.tolist() for row in mat],
                               type=pa.list_(pa.float32()))
                idx = tbl.column_names.index(col)
                tbl = tbl.set_column(idx, col, lst)

            pq.write_table(tbl, dst_f, compression="snappy")
            n_files += 1
            n_frames += tbl.num_rows

    print("  data: %d 个 parquet, 保留 %d 帧, 维度 %s -> %d"
          % (n_files, n_frames, src_dims, KEEP_DIM))
    if n_dropped:
        print("  丢帧 %s, 共丢 %d 帧 (%.1f%%)"
              % (drop_desc, n_dropped,
                 100.0 * n_dropped / max(n_frames + n_dropped, 1)))
    if n_zero_fixed:
        print("  另外修掉了 %d 个全零无效帧" % n_zero_fixed)

    # ---------- 视频：符号链接 ----------
    os.symlink(os.path.abspath(os.path.join(src, "videos")),
               os.path.join(dst, "videos"))

    # ---------- info.json ----------
    names = info["features"]["observation.state"].get("names")
    new_names = names[:KEEP_DIM] if names else None
    for key in ("observation.state", "action"):
        if key in info["features"]:
            info["features"][key]["shape"] = [KEEP_DIM]
            if new_names:
                info["features"][key]["names"] = list(new_names)
    info["video_path"] = ("videos/{video_key}/chunk-{episode_chunk:03d}/"
                          "file-{episode_index:03d}.mp4")
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["total_frames"] = int(n_frames)
    with open(os.path.join(meta_dst, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    print("  info.json: video_path -> %s" % info["video_path"])

    # ---------- stats.json ----------
    stats_src = os.path.join(meta_src, "stats.json")
    if os.path.isfile(stats_src):
        with open(stats_src, encoding="utf-8") as f:
            stats = json.load(f)
        for key in ("observation.state", "action"):
            if key in stats and isinstance(stats[key], dict):
                stats[key] = trim_stats_dict(stats[key])
        with open(os.path.join(meta_dst, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print("  stats.json: 已裁到 %d 维" % KEEP_DIM)

    # ---------- tasks.parquet ----------
    task_texts = {}
    tp = os.path.join(meta_src, "tasks.parquet")
    if os.path.isfile(tp):
        tdf = pd.read_parquet(tp)
        for pos, (idx, row) in enumerate(tdf.iterrows()):
            ti = row.get("task_index", pos)
            try:
                ti = int(ti)
            except Exception:
                ti = pos
            txt = ""
            for k in ("task", "tasks", "__index_level_0__"):
                try:
                    if k in row.index and row[k] is not None:
                        txt = str(row[k])
                        break
                except Exception:
                    pass
            if not txt:
                txt = str(idx)
            task_texts[ti] = clean_text(txt)
    if not task_texts:
        task_texts = {0: ""}

    out_tasks = pd.DataFrame({
        "task_index": sorted(task_texts),
        "task": [task_texts[k] for k in sorted(task_texts)],
    }).set_index("task_index")
    out_tasks.to_parquet(os.path.join(meta_dst, "tasks.parquet"))
    for k in sorted(task_texts):
        print("  task[%d] = %r" % (k, task_texts[k]))

    # ---------- meta/episodes ----------
    ep_dst_dir = os.path.join(meta_dst, "episodes")
    for dirpath, _, filenames in os.walk(os.path.join(meta_src, "episodes")):
        rel = os.path.relpath(dirpath, os.path.join(meta_src, "episodes"))
        out_dir = os.path.join(ep_dst_dir, rel) if rel != "." else ep_dst_dir
        os.makedirs(out_dir, exist_ok=True)
        for fn in sorted(filenames):
            if not fn.endswith(".parquet"):
                continue
            t = pq.read_table(os.path.join(dirpath, fn))
            cols = list(t.column_names)
            for col in cols:
                if not col.startswith("stats/"):
                    continue
                base = col.split("/", 1)[1]
                if base not in ("observation.state", "action"):
                    continue
                struct = t[col]
                new_rows = []
                for x in struct:
                    d = x.as_py() or {}
                    new_rows.append(trim_stats_dict(d))
                t = t.set_column(cols.index(col), col, pa.array(new_rows))
            # 同步 length / from_timestamp（加载器不依赖，但保持元数据自洽）
            if "episode_index" in t.column_names:
                ei = np.asarray(t["episode_index"].to_pylist(), dtype=np.int64)
                if isinstance(drop_first, int):
                    d = np.full(len(ei), max(drop_first, 0), dtype=np.int64)
                else:
                    d = np.array([ep_drop.get(int(e), 0) for e in ei],
                                 dtype=np.int64)
                if "length" in t.column_names:
                    arr = np.asarray(t["length"].to_pylist(), dtype=np.int64) - d
                    t = t.set_column(t.column_names.index("length"), "length",
                                     pa.array(np.maximum(arr, 0)))
                fps = float(info.get("fps", 30) or 30)
                for col in list(t.column_names):
                    if col.endswith("/from_timestamp"):
                        arr = np.asarray(t[col].to_pylist(), dtype=np.float64)
                        t = t.set_column(t.column_names.index(col), col,
                                         pa.array(arr + d / fps))
            pq.write_table(t, os.path.join(out_dir, fn), compression="snappy")
    print("  meta/episodes: 已同步裁到 %d 维" % KEEP_DIM)

    # ---------- 校验：视频路径能否拼出来 ----------
    first_ep = int(ep_df["episode_index"].min())
    chunks_size = int(info.get("chunks_size", 1000))
    for vk in [k for k in info["features"] if k.startswith("observation.images.")]:
        p = os.path.join(dst, info["video_path"].format(
            video_key=vk, episode_chunk=first_ep // chunks_size,
            episode_index=first_ep))
        if not os.path.exists(p):
            print("  [FAIL] 视频路径拼不出来: %s" % p)
            return False
    print("  视频路径校验: OK")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/lab/tron_ws/datasets_raw")
    ap.add_argument("--dst", default="/home/lab/tron_ws/FluxVLA/datasets/RealRobot_Tron2_lerobot")
    ap.add_argument("--drop-first", default="auto",
                    help="丢掉每条 episode 开头相机未出图的帧数；"
                         "'auto'=逐段扫描自适应（默认），或给整数固定丢弃")
    args = ap.parse_args()

    try:
        drop_first = int(args.drop_first)
    except (TypeError, ValueError):
        drop_first = "auto"

    os.makedirs(args.dst, exist_ok=True)
    names = sorted(d for d in os.listdir(args.src)
                   if os.path.isfile(os.path.join(args.src, d, "meta", "info.json")))
    if not names:
        print("在 %s 下没找到任何数据集" % args.src)
        return 1

    print("待处理 %d 个数据集: %s" % (len(names), names))
    ok_list, bad_list = [], []
    for d in names:
        try:
            if process_dataset(os.path.join(args.src, d), os.path.join(args.dst, d),
                               drop_first=args.drop_first):
                ok_list.append(d)
            else:
                bad_list.append(d)
        except Exception as e:
            import traceback
            print("  [EXCEPTION] %s: %s" % (type(e).__name__, e))
            traceback.print_exc()
            bad_list.append(d)

    print()
    print("=" * 74)
    print("完成: 成功 %d, 失败 %d" % (len(ok_list), len(bad_list)))
    for d in ok_list:
        print("  [OK]   %s" % d)
    for d in bad_list:
        print("  [FAIL] %s" % d)
    print()
    print("输出目录: %s" % args.dst)
    print("把下面这些路径填进 config 的 data_root_path 列表:")
    for d in ok_list:
        print("    './datasets/RealRobot_Tron2_lerobot/%s'," % d)
    return 0 if not bad_list else 1


if __name__ == "__main__":
    sys.exit(main())
