#!/bin/bash
# =============================================================================
#  TRON2 三按钮（红/黑/绿）LoRA 训练 —— 端到端（支持多卡）
#
#  只训练三个按钮任务，不含拧旋钮。流程：
#    1) 校验数据集目录里恰好 3 个按钮子集
#    2) 重算归一化统计量 + 退化维修正 -> tron2_stats_armsymmetric.json
#    3) 【就地打补丁】更新 config 的 data_root_path / batch / accum / lr / 步数
#    4) torchrun 启动训练（--nproc-per-node $NPROC）
#
#  ⚠️ 第 3 步刻意【不】调用 make_tron2_lora_config.py：
#     那个生成器从厂商基配重新生成，实测会冲掉本仓库手改的关键修复
#     （dataset_statistics_path 被换成 auto_compute_statistics；
#      DenormalizeTron2Action + state_permutation 退回 DenormalizeDeltaAction，
#      导致机器人 18 维动作展开失效）。所以这里只做定点替换。
#
#  ⚠️ 单卡 micro-batch 上限为 3（4090D 24G，4 会 OOM）。要更大的有效 batch，
#     只能靠 GPU 数 x grad_accum。有效 batch = BATCH_SIZE * NPROC * GRAD_ACCUM。
#
#  用法:
#    bash scripts/train_tron2_buttons.sh
#    NPROC=2 GRAD_ACCUM=16 LR=1e-4 MAX_STEPS=25000 bash scripts/train_tron2_buttons.sh
#    DRY_RUN=1 SKIP_STATS=1 bash scripts/train_tron2_buttons.sh   # 只演练前 3 步
#
#  可覆盖的环境变量:
#    NPROC        GPU 数 / torchrun --nproc-per-node（默认 2）
#    BATCH_SIZE   单卡 micro-batch（默认 3；24G 上限）
#    GRAD_ACCUM   梯度累积步数（默认 16）
#    LR           optimizer 峰值学习率（默认 1e-4）
#    MIN_LR       余弦退火终点（默认 1e-6）
#    MAX_STEPS    总训练步数（默认 25000）
#    SAVE_ITER    存档间隔（默认 1000；必须整除 MAX_STEPS，见 guide §7.6）
#    WORK_DIR     训练输出目录（默认 work_dirs/tron2_buttons_v2）
#    DRY_RUN=1    只跑到第 3 步，不启动训练
#    SKIP_STATS=1 跳过第 2 步（统计量已算过时用）
# =============================================================================
set -euo pipefail

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FV="$BASE"
PY=${FLUXVLA_PY:-/home/lab/miniconda3/envs/fluxvla/bin/python3}
TORCHRUN=${FLUXVLA_TORCHRUN:-/home/lab/miniconda3/envs/fluxvla/bin/torchrun}

DATA="$FV/datasets/RealRobot_Tron2_lerobot"
STATS="$DATA/tron2_stats_armsymmetric.json"
CONFIG="$FV/configs/pi05/pi05_paligemma_tron2_cabinet_lora.py"

NPROC=${NPROC:-2}
BATCH_SIZE=${BATCH_SIZE:-3}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-25000}
SAVE_ITER=${SAVE_ITER:-1000}
WORK_DIR=${WORK_DIR:-work_dirs/tron2_buttons_v2}
DRY_RUN=${DRY_RUN:-0}
SKIP_STATS=${SKIP_STATS:-0}

fail() { echo "[FAIL] $*" >&2; exit 1; }

[ -x "$PY" ] || fail "找不到 python: $PY"
[ -x "$TORCHRUN" ] || fail "找不到 torchrun: $TORCHRUN"
[ -f "$CONFIG" ] || fail "找不到训练 config: $CONFIG"
[ $((MAX_STEPS % SAVE_ITER)) -eq 0 ] || \
  fail "MAX_STEPS($MAX_STEPS) 不能被 SAVE_ITER($SAVE_ITER) 整除，最后一步不落盘（guide §7.6）"

# --- 1) 数据集目录校验 ------------------------------------------------------
echo "=== [1/4] 校验数据集目录 ==="
"$PY" - "$DATA" <<'PYCHECK'
import os, sys
root = sys.argv[1]
subs = sorted(d for d in os.listdir(root)
              if os.path.isfile(os.path.join(root, d, 'meta', 'info.json')))
print('发现子集 (%d):' % len(subs))
for s in subs:
    print('   -', s)
if len(subs) != 3:
    raise SystemExit('[FAIL] 期望恰好 3 个按钮子集。'
                     '\n       请把非按钮子集移到 %s/_archive/ 再跑。' % root)
PYCHECK

# --- 2) 统计量 --------------------------------------------------------------
if [ "$SKIP_STATS" = "1" ] && [ -f "$STATS" ]; then
  echo; echo "=== [2/4] 跳过统计量重算（SKIP_STATS=1，已有 $STATS）==="
else
  echo; echo "=== [2/4] 重算并修正归一化统计量 ==="
  "$PY" "$BASE/scripts/make_tron2_button_stats.py" \
    --fluxvla "$FV" --data-dir "$DATA" --out "$STATS"
fi

# --- 3) 就地打补丁 ----------------------------------------------------------
echo; echo "=== [3/4] 就地更新 config (保留所有手改修复) ==="
cp "$CONFIG" "$CONFIG.bak"
"$PY" - "$CONFIG" "$DATA" "$BATCH_SIZE" "$GRAD_ACCUM" "$MAX_STEPS" \
         "$SAVE_ITER" "$LR" "$MIN_LR" <<'PYPATCH'
import os, re, sys

(cfg, data, batch_size, grad_accum, max_steps, save_iter, lr, min_lr) = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
    sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
src = open(cfg, encoding='utf-8').read()

subs = sorted(d for d in os.listdir(data)
              if os.path.isfile(os.path.join(data, d, 'meta', 'info.json')))


def set_field(src, pattern, value, tag):
    hits = re.findall(pattern, src)
    if len(hits) != 1:
        raise SystemExit('[FAIL] %s 匹配 %d 处（应为 1）: %r'
                         % (tag, len(hits), pattern))
    return src.replace(hits[0], value)


# data_root_path 块
m = re.search(
    r"(data_root_path=\[\s*# noqa: E501\n)"
    r"((?:\s+'\./datasets/RealRobot_Tron2_lerobot/[^']+',\s*# noqa: E501\n)+)"
    r"(\s*\],)", src)
if not m:
    raise SystemExit('[FAIL] 找不到 data_root_path 块')
indent = re.match(r"\s*", m.group(2)).group(0)
block = m.group(1) + ''.join(
    "%s'./datasets/RealRobot_Tron2_lerobot/%s',  # noqa: E501\n" % (indent, s)
    for s in subs) + m.group(3)
src = src[:m.start()] + block + src[m.end():]

fields = [
    ('per_device_batch_size', r"    per_device_batch_size=\d+,",
     "    per_device_batch_size=%s," % batch_size),
    ('grad_accumulation_steps', r"    grad_accumulation_steps=\d+,",
     "    grad_accumulation_steps=%s," % grad_accum),
    ('max_steps', r"    max_steps=\d+,", "    max_steps=%s," % max_steps),
    ('save_iter_interval', r"    save_iter_interval=\d+,",
     "    save_iter_interval=%s," % save_iter),
    ('decay_steps', r"        decay_steps=\d+,", "        decay_steps=%s," % max_steps),
    ('lr', r"        lr=[0-9.eE+-]+,", "        lr=%s," % lr),
    ('min_lr', r"        min_lr=[0-9.eE+-]+\),", "        min_lr=%s)," % min_lr),
]
for tag, pat, val in fields:
    src = set_field(src, pat, val, tag)

open(cfg, 'w', encoding='utf-8').write(src)
print('[ok] data_root_path -> %d 个子集' % len(subs))
print('[ok] per_device_batch_size=%s  grad_accumulation_steps=%s'
      % (batch_size, grad_accum))
print('[ok] max_steps=%s  save_iter_interval=%s  decay_steps=%s'
      % (max_steps, save_iter, max_steps))
print('[ok] lr=%s  min_lr=%s' % (lr, min_lr))
PYPATCH

"$PY" -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read()); print('[ok] config 语法 OK')" "$CONFIG"

EFF_BATCH=$((BATCH_SIZE * NPROC * GRAD_ACCUM))
echo
echo "有效 batch = $BATCH_SIZE(per-dev) x $NPROC(GPU) x $GRAD_ACCUM(accum) = $EFF_BATCH"

if [ "$DRY_RUN" = "1" ]; then
  echo; echo "DRY_RUN=1，停在第 3 步。备份在 $CONFIG.bak"; exit 0
fi

# --- 4) 训练 ---------------------------------------------------------------
echo; echo "=== [4/4] 启动训练 -> $WORK_DIR  (nproc=$NPROC) ==="
cd "$FV"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "监控: tail -1 $WORK_DIR/pi05_paligemma_tron2_cabinet_lora_*.jsonl"

exec "$TORCHRUN" --standalone --nnodes 1 --nproc-per-node "$NPROC" \
  scripts/train.py --config "configs/pi05/pi05_paligemma_tron2_cabinet_lora.py" \
  --work-dir "$WORK_DIR"
