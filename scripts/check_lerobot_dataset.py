#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 数据集体检脚本 —— 兼容 v2.1 / v3.0 两种目录布局

用法:
    python3 check_lerobot_dataset.py <数据集目录>                # 全量体检
    python3 check_lerobot_dataset.py <数据集目录> --detail 5      # 逐条打印前 5 个 episode
    python3 check_lerobot_dataset.py <数据集目录> --json out.json # 结论另存 JSON

设计要点:
  * 不猜目录布局 —— 直接递归扫所有 .parquet，靠 episode_index 列分组。
    因此 v3.0 的 data/chunk-000/file-*.parquet 和 v2.1 的
    data/train/episode_*.parquet 都能吃。
  * 只读需要的列，避免把视频/大字段读进内存。
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

try:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as e:
    print("缺少依赖: %s" % e)
    print("请先执行:  pip3 install pandas pyarrow numpy")
    sys.exit(2)

PROBLEMS = []          # (level, title, detail)
STATS = {}             # 汇总，便于 --json 输出
HAS_MP4 = 0
VIDEO_KEYS = []


def record(level, title, detail=""):
    PROBLEMS.append((level, title, detail))


def hr(ch="-", n=78):
    print(ch * n)


def find_roots(path):
    """返回 (数据集根列表, 提示信息)"""
    if os.path.isfile(os.path.join(path, "meta", "info.json")):
        return [path], None
    cands = []
    if os.path.isdir(path):
        for d in sorted(os.listdir(path)):
            sub = os.path.join(path, d)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "meta", "info.json")):
                cands.append(sub)
    if cands:
        return cands, "自动下钻，发现 %d 个数据集" % len(cands)
    # 没有 info.json，也允许硬扫（可能导出不完整）
    return [path], None


def scan_parquet(root):
    """只扫 data/ 下的 parquet（必须避开 meta/tasks.parquet 与 meta/episodes/*.parquet）"""
    out = {}
    data_root = os.path.join(root, "data")
    base = data_root if os.path.isdir(data_root) else root
    for dirpath, dirnames, filenames in os.walk(base):
        # 跳过可能存在的备份/隐藏目录
        if os.sep + "." in dirpath:
            continue
        for fn in filenames:
            if fn.endswith(".parquet"):
                full = os.path.join(dirpath, fn)
                out[full] = os.path.relpath(full, root)
    return out


def count_videos(root):
    """统计 mp4 分布: {video_key: 数量}"""
    per_key = defaultdict(int)
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".mp4"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            parts = rel.split(os.sep)
            # v3.0: videos/chunk-000/<video_key>/file-XXX.mp4
            # v2.1: data/video/episode_XXXXXX/<video_key>.mp4
            key = None
            for p in parts:
                if p.startswith("observation.images."):
                    key = p
                    break
            if key is None:
                if parts and parts[-1].startswith("observation.images."):
                    key = parts[-1].replace(".mp4", "")
                else:
                    key = parts[-2] if len(parts) >= 2 else "?"
            per_key[key] += 1
    return per_key


def load_frames(files):
    """读取所有 parquet 到单个 DataFrame（只取需要的列）"""
    frames = []
    used_cols = None
    for full, rel in sorted(files.items()):
        try:
            schema = pq.read_schema(full)
        except Exception as e:
            record("FAIL", "parquet 无法读取: %s" % rel, str(e))
            continue
        names = set(schema.names)
        want = [c for c in ("observation.state", "action", "actions",
                            "timestamp", "episode_index", "frame_index",
                            "index", "task_index", "task",
                            "language_instruction") if c in names]
        if not want:
            record("FAIL", "parquet 里没有可识别字段: %s" % rel,
                   "字段: %s" % ", ".join(sorted(names))[:200])
            continue
        if used_cols is None:
            used_cols = want
        try:
            df = pd.read_parquet(full, columns=want)
        except Exception as e:
            record("FAIL", "parquet 读取失败: %s" % rel, str(e))
            continue
        df["_src_file"] = rel
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def to_mat(series):
    """把 list/array 列转成 (N, D) float32 矩阵"""
    arrs = []
    for x in series.values:
        if x is None:
            arrs.append(None)
            continue
        a = np.asarray(x, dtype=np.float32).ravel()
        arrs.append(a)
    valid = [a for a in arrs if a is not None and a.size]
    if not valid:
        return None
    dim = max(a.size for a in valid)
    out = np.full((len(arrs), dim), np.nan, dtype=np.float32)
    for i, a in enumerate(arrs):
        if a is not None and a.size:
            out[i, :a.size] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="数据集目录（可以是根目录，也可以是包含多个数据集的上级目录）")
    ap.add_argument("--detail", type=int, default=0, help="逐条打印前 N 个 episode")
    ap.add_argument("--json", default=None, help="把结论写入 JSON 文件")
    args = ap.parse_args()

    roots, note = find_roots(args.path)

    for root in roots:
        del PROBLEMS[:]          # 每个数据集独立计数，避免告警重复堆积
        print("=" * 78)
        print("数据集根目录 : %s" % os.path.abspath(root))
        if note:
            print("提示         : %s" % note)
        print("=" * 78)

        # ---------- meta ----------
        info = {}
        info_path = os.path.join(root, "meta", "info.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                print("meta 目录    : %s" % os.path.join(root, "meta"))
            except Exception as e:
                record("FAIL", "info.json 解析失败", str(e))
        else:
            print("meta 目录    : 未找到 info.json")
            record("FAIL", "找不到 meta/info.json",
                   "目录里没有 info.json，先确认导出是否成功、路径对不对")

        ver = info.get("codebase_version", "未知（info.json 里没有该字段）")
        print("codebase 版本: %s" % ver)
        print("robot_type   : %s" % info.get("robot_type", "未知"))
        STATS["codebase_version"] = ver
        STATS["robot_type"] = info.get("robot_type")

        # ---------- 文件 ----------
        files = scan_parquet(root)
        vids = count_videos(root)
        n_mp4 = sum(vids.values())
        print("数据文件数   : %d 个 parquet" % len(files))
        print("视频文件数   : %d 个 mp4" % n_mp4)
        STATS["parquet"] = len(files)
        STATS["mp4"] = n_mp4
        STATS["video_keys"] = dict(vids)

        if not files:
            record("FAIL", "找不到数据 parquet",
                   "目录里没有可读的数据文件，先确认导出是否成功、路径对不对")
            summarize()
            continue

        # ---------- features ----------
        feats = info.get("features")
        cam_names = []
        state_dim = None
        if isinstance(feats, dict) and feats:
            print("-" * 78)
            print("features schema")
            for k, v in feats.items():
                shape = v.get("shape") if isinstance(v, dict) else None
                dt = v.get("dtype") if isinstance(v, dict) else None
                print("  %-46s %-10s %s" % (k, dt, shape))
                if k.startswith("observation.images."):
                    cam_names.append(k)
                if k in ("observation.state",) and shape:
                    state_dim = shape[0] if isinstance(shape, (list, tuple)) else shape
            STATS["cameras"] = cam_names
            STATS["state_dim_declared"] = state_dim
        else:
            record("WARN", "info.json 里没有 features 字段",
                   "无法确认字段 schema，只能靠 parquet 内容推断")

        # 夹爪维度索引：优先用 info.json 的 names 定位（兼容 14/16/21 维等布局）
        grip_idx = []
        try:
            st_names = (feats or {}).get("observation.state", {}).get("names")
            if st_names:
                grip_idx = [i for i, n in enumerate(st_names)
                            if "gripper" in str(n).lower()]
        except Exception:
            pass
        if not grip_idx:
            grip_idx = [6, 13]
        print("夹爪维度索引 : %s" % grip_idx)

        fps = info.get("fps")
        if fps:
            print("fps          : %s" % fps)
        print("总 episode   : %s (info.json)" % info.get("total_episodes", "未声明"))
        print("总帧数       : %s (info.json)" % info.get("total_frames", "未声明"))

        # ---------- 读数据 ----------
        print("-" * 78)
        print("读取 parquet ...")
        frames = load_frames(files)
        if frames is None or frames.empty:
            record("FAIL", "读取不到任何数据行", "parquet 全部为空或不可读")
            summarize()
            continue
        print("  -> %d 行, %d 列" % (len(frames), len(frames.columns)))

        # episode 分组
        if "episode_index" in frames.columns:
            ep_col = frames["episode_index"]
            if ep_col.isna().all():
                ep_col = pd.Series(np.zeros(len(frames), dtype=np.int64), name="episode_index")
                record("WARN", "episode_index 列全为空",
                       "已按单个 episode 处理，无法判断是否真的切分了 episode")
        else:
            ep_col = pd.Series(np.zeros(len(frames), dtype=np.int64), name="episode_index")
            record("WARN", "没有 episode_index 列",
                   "无法区分 episode，只能整体当一个 episode 统计")

        state_col = "observation.state" if "observation.state" in frames.columns else None
        act_col = "action" if "action" in frames.columns else (
            "actions" if "actions" in frames.columns else None)
        if state_col is None:
            record("FAIL", "缺少 observation.state 列", "数据集里没有 state，无法训练")
        if act_col is None:
            record("FAIL", "缺少 action 列", "数据集里没有 action，无法训练（无法学习动作）")

        S = to_mat(frames[state_col]) if state_col else None
        A = to_mat(frames[act_col]) if act_col else None

        # 实际 state 维度
        if S is not None:
            print("-" * 78)
            print("state 维度   : %d" % S.shape[1])
            STATS["state_dim"] = int(S.shape[1])
            if state_dim and int(S.shape[1]) != int(state_dim):
                record("WARN", "state 维度与 info.json 不符",
                       "info.json=%s, 实际=%d" % (state_dim, S.shape[1]))
            if S.shape[1] == 14:
                print("             -> 14 维，正好 = 左臂7 + 右臂7（符合 FluxVLA 约定）")
            elif S.shape[1] == 16:
                print("             -> 16 维，FluxVLA 会自动转成 14 维，但注意映射是否正确")
            else:
                record("WARN", "state 维度不是 14 也不是 16",
                       "实际 %d 维，FluxVLA 的 TRON2 配置是按 14 维写的，需要核对" % S.shape[1])
        if A is not None:
            print("action 维度  : %d" % A.shape[1])
            STATS["action_dim"] = int(A.shape[1])
            if S is not None and A.shape[1] != S.shape[1]:
                record("WARN", "action 与 state 维度不一致",
                       "state=%d, action=%d，通常应该一致" % (S.shape[1], A.shape[1]))

        # ---------- episode 统计 ----------
        print("-" * 78)
        groups = list(frames.groupby(ep_col, sort=True))
        ep_lens = []
        ep_rows = []
        for ep_key, g in groups:
            n = len(g)
            ep_lens.append(n)
            dur = None
            if "timestamp" in g.columns and len(g) > 1:
                ts = g["timestamp"].values.astype(np.float64)
                dur = float(ts.max() - ts.min())
            task = ""
            if "task" in g.columns:
                vals = [v for v in g["task"].dropna().unique().tolist() if str(v)]
                task = str(vals[0]) if vals else ""
            ep_rows.append({"episode": ep_key, "frames": n, "duration": dur, "task": task})

        ep_lens = np.array(ep_lens, dtype=np.float64)
        print("episode 数量 : %d" % len(ep_rows))
        print("总帧数       : %d" % len(frames))
        if fps:
            print("总时长       : %.1f 秒" % (len(frames) / float(fps)))
        print("帧数  min/中位/mean/max : %.0f / %.0f / %.1f / %.0f"
              % (ep_lens.min(), np.median(ep_lens), ep_lens.mean(), ep_lens.max()))
        STATS["episodes"] = len(ep_rows)
        STATS["total_frames"] = int(len(frames))
        STATS["ep_len_min"] = int(ep_lens.min())
        STATS["ep_len_max"] = int(ep_lens.max())

        # v3.0: meta/episodes/*.parquet 里的 episode_success
        eps_files = sorted(glob.glob(os.path.join(root, "meta", "episodes",
                                                  "**", "*.parquet"), recursive=True))
        if eps_files:
            try:
                ed = pd.concat([pd.read_parquet(p) for p in eps_files], ignore_index=True)
                if "episode_success" in ed.columns:
                    vc = ed["episode_success"].value_counts().to_dict()
                    print("episode_success: %s" % vc)
                    STATS["episode_success"] = {str(k): int(v) for k, v in vc.items()}
                    ok = sum(int(v) for k, v in vc.items()
                             if str(k).lower() in ("success", "true", "1"))
                    if ok < len(ed):
                        record("WARN", "有 %d 条 episode 未标记为 success"
                               % (len(ed) - ok),
                               "失败/中断的 episode 会污染训练数据，建议导出前剔除")
            except Exception as e:
                record("WARN", "meta/episodes 读取失败", str(e))

        # ---------- 各项检查 ----------
        # 1) 过短 episode
        short = [r for r in ep_rows if r["frames"] < 10]
        if short:
            record("WARN", "有 %d 个 episode 帧数 < 10" % len(short),
                   "过短的 episode 学不到有效动作；举例: %s"
                   % ", ".join("ep%s=%d帧" % (r["episode"], r["frames"]) for r in short[:6]))
        if len(ep_lens) > 1 and ep_lens.mean() > 0:
            cv = ep_lens.std() / ep_lens.mean()
            if cv > 0.8:
                record("WARN", "episode 长度差异过大 (变异系数 %.2f)" % cv,
                       "min=%d max=%d。差异过大往往说明切分边界不对或任务不一致"
                       % (ep_lens.min(), ep_lens.max()))

        # 2) 夹爪维度（索引由 info.json 的 names 决定）
        if S is not None and grip_idx:
            labels = ["左", "右", "夹爪3", "夹爪4"]
            for k, idx in enumerate(grip_idx):
                if idx >= S.shape[1]:
                    continue
                nm = labels[k] if k < len(labels) else ("第%d个" % (k + 1))
                col = S[:, idx]
                col = col[~np.isnan(col)]
                if col.size == 0:
                    continue
                span = float(col.max() - col.min())
                if span < 1e-6:
                    record("WARN", "%s夹爪(第%d维)全程不变 (值=%.4f)" % (nm, idx, col.mean()),
                           "夹爪可能没校零、没记录，或该任务确实没用夹爪。"
                           "夹爪是 action 的一维，恒定值会让这一维学不到东西")
                else:
                    print("夹爪 第%-2d维(%s) : %9.4f ~ %9.4f  span=%.4f  (有变化 OK)"
                          % (idx, nm, col.min(), col.max(), span))

        # 3) 左右臂活动量（只统计 episode 内部帧间变化，避免跨 episode 跳变污染）
        if S is not None and S.shape[1] >= 14:
            idxs = np.asarray(ep_col.values)
            same_ep = np.zeros(len(S), dtype=bool)
            if len(S) > 1:
                same_ep[1:] = idxs[1:] == idxs[:-1]
            for nm, sl in (("左臂", slice(0, 7)), ("右臂", slice(7, 14))):
                sub = S[:, sl]
                rng = float(np.nanmax(sub) - np.nanmin(sub))
                if same_ep[1:].sum() > 1:
                    d = np.abs(np.diff(sub, axis=0))[same_ep[1:]]
                    act = float(np.nanmean(d))
                else:
                    act = 0.0
                print("%s  : 帧间步进 %.5f, 总幅度 %.3f (跨 %d 条 episode)"
                      % (nm, act, rng, len(groups)))
                if rng < 1e-3:
                    record("WARN", "%s 全程几乎没动 (幅度 %.5f)" % (nm, rng),
                           "这一侧可能根本没被控制，或录制时选错了关节来源")

        # 4) 时间戳
        if "timestamp" in frames.columns:
            bad_gap = 0
            worst = 0.0
            ep_gaps = []
            for ep_key, g in groups:
                ts = np.sort(g["timestamp"].values.astype(np.float64))
                if ts.size < 3:
                    continue
                d = np.diff(ts)
                if d.size == 0:
                    continue
                med = float(np.median(d))
                if med <= 0:
                    continue
                ep_gaps.append(med)
                m = float(d.max())
                if m > med * 3 + 1e-6:
                    bad_gap += 1
                    worst = max(worst, m)
            if ep_gaps:
                allmed = float(np.median(ep_gaps))
                print("时间戳间隔   : 中位 %.4f s (≈ %.1f Hz)" % (allmed, 1.0 / allmed if allmed else 0))
                STATS["median_dt"] = allmed
                if fps and abs(1.0 / allmed - float(fps)) > float(fps) * 0.15:
                    record("WARN", "实测帧率与 info.json 的 fps 不符",
                           "实测 ≈%.1f Hz，info.json fps=%s" % (1.0 / allmed, fps))
            if bad_gap:
                record("WARN", "有 %d 个 episode 存在时间戳跳变（最大 %.3f s）" % (bad_gap, worst),
                       "可能是掉帧或暂停。时间戳不连续会污染时间对齐，建议检查")
            # 是否从 0 开始 / 是否单调
            neg = 0
            mono_bad = 0
            for ep_key, g in groups:
                ts = g["timestamp"].values.astype(np.float64)
                if ts.size > 1 and np.any(np.diff(ts) < -1e-6):
                    mono_bad += 1
                if ts.size and ts.min() < -1e-9:
                    neg += 1
            if mono_bad:
                record("WARN", "有 %d 个 episode 的时间戳不是单调递增" % mono_bad,
                       "同一 episode 内时间戳应单调递增")

        # 5) action 与 state 的时间对齐
        if S is not None and A is not None and S.shape == A.shape:
            diff_same = float(np.nanmean(np.abs(A - S)))
            base = float(np.nanmean(np.abs(S))) + 1e-9
            rel_same = diff_same / base
            # action[t] vs state[t+1]，同一 episode 内才比较
            rel_next = float("nan")
            fr = (frames["frame_index"].values if "frame_index" in frames.columns
                  else np.arange(len(frames)))
            order = np.lexsort((fr, np.asarray(ep_col.values)))
            Ss, As, Es = S[order], A[order], np.asarray(ep_col.values)[order]
            nxt = [np.abs(As[i] - Ss[i + 1])
                   for i in range(len(Es) - 1) if Es[i] == Es[i + 1]]
            if nxt:
                rel_next = float(np.nanmean(np.vstack(nxt))) / base
                print("action 对齐  : 同帧相对差 %.4f   action[t] vs state[t+1] 相对差 %.4f"
                      % (rel_same, rel_next))
            else:
                print("action 对齐  : 同帧相对差 %.4f" % rel_same)
            STATS["action_same_frame_rel"] = rel_same
            STATS["action_next_frame_rel"] = rel_next
            if rel_same < 0.005:
                record("FAIL", "action 几乎等于同帧 state（相对差 %.4f）" % rel_same,
                       "动作标签=当前状态，模型只会学到恒等映射。"
                       "常见原因：采集时 action 直接拷贝了 state")
            elif rel_next == rel_next and rel_next < rel_same:
                print("             -> action[t] 更贴近 state[t+1]，时间对齐正常")

        # 6) 任务标签
        print("-" * 78)
        tasks = {}
        if "task_index" in frames.columns and "task" in frames.columns:
            for ti, g in frames.groupby("task_index"):
                vals = [v for v in g["task"].dropna().unique().tolist() if str(v)]
                tasks[int(ti)] = vals[0] if vals else ""
        elif os.path.isfile(os.path.join(root, "meta", "tasks.parquet")):
            try:
                tdf = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
                for pos, (idx, row) in enumerate(tdf.iterrows()):
                    ti = row.get("task_index", pos)
                    try:
                        ti = int(ti)
                    except Exception:
                        ti = pos
                    txt = ""
                    for key in ("task", "tasks", "__index_level_0__"):
                        try:
                            if key in row.index and row[key] is not None:
                                txt = str(row[key])
                                break
                        except Exception:
                            pass
                    if not txt:
                        txt = str(idx)
                    tasks[ti] = txt
            except Exception as e:
                record("WARN", "tasks.parquet 读取失败", str(e))
        elif os.path.isfile(os.path.join(root, "meta", "tasks.jsonl")):
            with open(os.path.join(root, "meta", "tasks.jsonl"), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                        tasks[int(o.get("task_index", len(tasks)))] = o.get("task", "")
                    except Exception:
                        pass
        elif os.path.isfile(os.path.join(root, "meta", "tasks.json")):
            try:
                with open(os.path.join(root, "meta", "tasks.json"), encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        tasks[int(k)] = v
            except Exception:
                pass
        if tasks:
            print("任务标签 (%d 个):" % len(tasks))
            for k in sorted(tasks):
                cnt = 0
                if "task_index" in frames.columns:
                    cnt = int((frames["task_index"] == k).sum())
                pct = 100.0 * cnt / len(frames) if len(frames) else 0
                print("  [%3d] %-52s %6d 帧 (%.1f%%)" % (k, str(tasks[k])[:52], cnt, pct))
            STATS["tasks"] = tasks
            if len(tasks) > 1:
                record("WARN", "这个数据集里有 %d 种任务标签" % len(tasks),
                       "一条 label 描述不了多种动作会糊掉训练标签。"
                       "建议一个子任务一个数据集，导出时分开填任务描述")
        else:
            record("WARN", "找不到任务标签 (tasks.jsonl / task 列)",
                   "没有 language label，VLA 训练时没有指令输入")

        # 零宽字符 / 首尾空白检查（数采软件常带入 U+200B）
        if tasks:
            bad_txt = {}
            for k, v in tasks.items():
                s = str(v)
                if ("\u200b" in s or "\ufeff" in s or "\u200e" in s
                        or s != s.strip()):
                    bad_txt[k] = s
            if bad_txt:
                record("WARN", "有 %d 个任务文本含不可见字符或首尾空白" % len(bad_txt),
                       "零宽空格/空白会污染 prompt token，导致同义指令被当成不同序列，"
                       "训练前必须清洗。例: %r" % list(bad_txt.values())[0])

        # 7) 视频数量核对
        if vids and cam_names:
            expect = len(ep_rows) * len(cam_names)
            if n_mp4 < expect:
                record("WARN", "视频数量不足",
                       "期望 %d 个 (episode %d × 相机 %d)，实际 %d 个"
                       % (expect, len(ep_rows), len(cam_names), n_mp4))
            else:
                print("视频核对     : OK (%d 个 >= 期望 %d)" % (n_mp4, expect))
        elif not vids:
            record("FAIL", "一个 mp4 都没有",
                   "VLA 需要视觉输入，没有视频就无法训练")

        # ---------- 逐条明细 ----------
        if args.detail > 0:
            print("-" * 78)
            print("前 %d 个 episode 明细" % args.detail)
            print("  %-8s %8s %10s  %s" % ("episode", "帧数", "时长(s)", "任务"))
            for r in ep_rows[: args.detail]:
                print("  %-8s %8d %10s  %s"
                      % (r["episode"], r["frames"],
                         ("%.2f" % r["duration"]) if r["duration"] is not None else "-",
                         str(r["task"])[:40]))

        summarize()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"stats": STATS, "problems": PROBLEMS}, f,
                      ensure_ascii=False, indent=2)
        print("结论已写入 %s" % args.json)


def summarize():
    print("=" * 78)
    print("体检结论")
    print("=" * 78)
    if not PROBLEMS:
        print("[PASS] 没发现问题，数据集看起来可用")
        hr("=")
        print()
        return
    order = {"FAIL": 0, "WARN": 1}
    for lv, title, detail in sorted(PROBLEMS, key=lambda x: order.get(x[0], 9)):
        print("[%s] %s" % (lv, title))
        if detail:
            print("        %s" % detail)
    hr("-")
    n_fail = sum(1 for p in PROBLEMS if p[0] == "FAIL")
    n_warn = sum(1 for p in PROBLEMS if p[0] == "WARN")
    if n_fail:
        print(">>> %d 个严重问题，先修这些问题再开量采集" % n_fail)
    else:
        print(">>> 没有严重问题，%d 个警告建议关注" % n_warn)
    hr("=")
    print()


if __name__ == "__main__":
    main()
