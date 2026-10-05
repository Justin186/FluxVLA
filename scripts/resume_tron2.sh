#!/bin/bash
# =============================================================================
#  TRON2 三按钮 LoRA —— 续训（从最新或指定检查点继续）
#
#  为什么需要这个脚本：
#    train_tron2_buttons.sh 只会**从零开始**训练，它最后是固定的
#    `exec torchrun ... --work-dir ...`，不接受 --resume-from。
#    续训必须自己拼长命令，容易把 step 号写错、把 .safetensors 当成 .pt 用。
#    这个脚本自动找最新的 step-*.pt 并把其它参数补齐。
#
#  用法:
#    bash scripts/resume_tron2.sh            # 自动取最新的 step-*.pt
#    bash scripts/resume_tron2.sh 4000       # 指定从 step 4000 续训
#    DRY_RUN=1 bash scripts/resume_tron2.sh  # 只打印命令，不启动
#
#  可覆盖的环境变量:
#    NPROC      GPU 数（默认 2；必须与出检查点时的卡数无关，但建议一致）
#    WORK_DIR   训练目录（默认 FluxVLA/work_dirs/tron2_buttons_v2）
#    LOG        日志文件（默认 tron_ws/logs/train_buttons_v2_resume.log，会覆盖）
#    DRY_RUN=1  只打印将执行的命令
#
#  ⚠️ 续训依赖两样东西，缺一不可：
#    1) checkpoints/step-XXXXXX-*.pt        含优化器/调度器状态
#    2) <WORK_DIR>/adapter_model.safetensors  LoRA 权重（resume 的真正来源）
#       原因：.pt 里存的是 merge_and_unload() 之后的**合并权重**，key 与
#       PEFT 包装的模型对不上；直接 load_state_dict(strict=False) 会静默丢弃
#       全部张量。详见 docs/pi05_tron2_cabinet_lora_training_config.md 第 13 章。
# =============================================================================
set -euo pipefail

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FV="$BASE"
TORCHRUN=${FLUXVLA_TORCHRUN:-/home/lab/miniconda3/envs/fluxvla/bin/torchrun}
CONFIG=configs/pi05/pi05_paligemma_tron2_cabinet_lora.py

NPROC=${NPROC:-2}
WORK_DIR=${WORK_DIR:-$FV/work_dirs/tron2_buttons_v2}
LOG=${LOG:-/home/lab/tron_ws/logs/train_buttons_v2_resume.log}
DRY_RUN=${DRY_RUN:-0}
STEP=${1:-}

fail() { echo "[FAIL] $*" >&2; exit 1; }

[ -x "$TORCHRUN" ] || fail "找不到 torchrun: $TORCHRUN"
CKPT_DIR="$WORK_DIR/checkpoints"
[ -d "$CKPT_DIR" ] || fail "找不到检查点目录: $CKPT_DIR"

# --- 1) 选检查点（一定要 .pt，不是 .safetensors）------------------------------
if [ -n "$STEP" ]; then
  CKPT=$(ls -1 "$CKPT_DIR"/step-"$(printf '%06d' "$STEP")"-*.pt 2>/dev/null | head -1) || true
  [ -n "$CKPT" ] || fail "找不到 step $STEP 的检查点（目录：$CKPT_DIR）"
else
  CKPT=$(ls -1t "$CKPT_DIR"/step-*.pt 2>/dev/null | head -1) || true
  [ -n "$CKPT" ] || fail "目录里没有 step-*.pt：$CKPT_DIR"
fi
echo "[ok] 续训检查点: $(basename "$CKPT")"

# --- 2) adapter 必须存在，否则 LoRA 权重恢复不了 -----------------------------
ADAPTER="$WORK_DIR/adapter_model.safetensors"
if [ -f "$ADAPTER" ]; then
  echo "[ok] LoRA adapter: $(basename "$ADAPTER") ($(stat -c %s "$ADAPTER") bytes)"
else
  fail "缺少 $ADAPTER —— 没有它 resume 无法恢复 LoRA 权重（见脚本头部说明）"
fi

# --- 3) 拒绝带残留进程启动（两个进程抢同一份检查点目录会互相覆盖）-----------
#     模式必须带上 config 名：这台机器上还有别的仓库在跑 scripts/train.py
#     （例如 ~/fc_ws/openpi），只匹配 "scripts/train.py" 会把它们也算进来。
if pgrep -f "scripts/train.py.*pi05_paligemma_tron2" >/dev/null 2>&1; then
  fail "已有本项目的训练进程在跑。先停： pkill -f 'scripts/train.py.*pi05_paligemma_tron2'"
fi

# --- 4) 启动 ----------------------------------------------------------------
cd "$FV"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=("$TORCHRUN" --standalone --nnodes 1 --nproc-per-node "$NPROC"
     scripts/train.py
     --config "$CONFIG"
     --work-dir "$WORK_DIR"
     --resume-from "$CKPT")

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] ${CMD[*]}"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
if [ -f "$LOG" ]; then
  cp -f "$LOG" "$LOG.prev"
  echo "[ok] 旧日志已备份为 $(basename "$LOG").prev"
fi

echo "[ok] 启动 nproc=$NPROC  log=$LOG"
setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &

sleep 6
echo "--- 进程 ---"
pgrep -af "scripts/train.py.*pi05_paligemma_tron2" | head -3
echo "--- 提示 ---"
echo "看进度: tr '\\r' '\\n' < $LOG | grep 'Global Step' | tail -1"
