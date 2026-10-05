#!/bin/bash
# 验证「梯度检查点 ON + 更大 batch」是否是更优组合。
# 上一轮扫描 gckpt 一直是 off；这轮把 ON 补上做对比。

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG=configs/pi05/pi05_paligemma_tron2_cabinet_lora.py
WD=work_dirs/sweep3
# 这个变量装的是 torchrun（名字沿用历史），自动探测，可用 FLUXVLA_TORCHRUN 覆盖
PY=${FLUXVLA_TORCHRUN:-$(dirname "$(command -v python3 2>/dev/null || echo /usr/bin/python3)")/torchrun}

rm -rf $WD
mkdir -p $WD

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one () {
    local NAME="$1"; local BS="$2"; local GA="$3"; local CK="$4"
    local LOG=/tmp/bc_${NAME}.log
    echo "=============== $NAME  (batch=$BS accum=$GA 有效=$((BS*GA)) gckpt=$CK) ==============="

    # 后台采样显存峰值
    ( for i in $(seq 1 200); do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
        sleep 1
      done ) > /tmp/bc_${NAME}.mem 2>/dev/null &
    local MEMPID=$!

    timeout 1500 $PY --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
        --config $CFG --work-dir $WD \
        --cfg-options \
            train_dataloader.per_device_batch_size=$BS \
            runner.grad_accumulation_steps=$GA \
            runner.enable_gradient_checkpointing=$CK \
            runner.max_steps=12 \
        > $LOG 2>&1

    kill $MEMPID 2>/dev/null

    if tr '\r' '\n' < $LOG | grep -q "OutOfMemoryError"; then
        echo "  ❌ OOM"
    else
        SIT=$(tr '\r' '\n' < $LOG | grep -oE "[0-9]+\.[0-9]+s/it" | tail -1)
        PEAK=$(sort -n /tmp/bc_${NAME}.mem 2>/dev/null | tail -1)
        echo "  ✅ ${SIT:-未取到}   峰值显存 ${PEAK:-?} MiB"
    fi
    echo
}

run_one "bs2_ckON"  2 4 true     # 基线：与上轮 bs2+gckpt OFF (1.75) 对比
run_one "bs4_ckON"  4 2 true     # gckpt ON 后 batch=4 能否装下
run_one "bs3_ckON"  3 3 true     # batch=3（有效 batch 9）

echo "=============== 结束 ==============="
