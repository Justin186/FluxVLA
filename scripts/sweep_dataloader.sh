#!/bin/bash
# 数据加载扫描：瓶颈在视频解码（GPU 利用率仅 43%）
# 固定 per_device_batch_size=2 / accum=4，逐个测 worker 数与视频解码后端

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG=configs/pi05/pi05_paligemma_tron2_cabinet_lora.py
WD=work_dirs/sweep2
# 这个变量装的是 torchrun（名字沿用历史），自动探测，可用 FLUXVLA_TORCHRUN 覆盖
PY=${FLUXVLA_TORCHRUN:-$(dirname "$(command -v python3 2>/dev/null || echo /usr/bin/python3)")/torchrun}

rm -rf $WD
mkdir -p $WD

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one () {
    local NAME="$1"; shift
    local LOG=/tmp/dl_${NAME}.log
    echo "=============== $NAME ==============="
    timeout 1500 $PY --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
        --config $CFG --work-dir $WD \
        --cfg-options \
            train_dataloader.per_device_batch_size=2 \
            runner.grad_accumulation_steps=4 \
            runner.enable_gradient_checkpointing=False \
            runner.max_steps=12 \
            "$@" \
        > $LOG 2>&1
    if tr '\r' '\n' < $LOG | grep -q "OutOfMemoryError"; then
        echo "  ❌ OOM"
    elif tr '\r' '\n' < $LOG | grep -qE "Error|Traceback"; then
        echo "  ❌ 报错:"
        tr '\r' '\n' < $LOG | grep -E "Error|Traceback" | tail -3 | cut -c1-150 | sed 's/^/     /'
    else
        SIT=$(tr '\r' '\n' < $LOG | grep -oE "[0-9]+\.[0-9]+s/it" | tail -1)
        echo "  ✅ ${SIT:-未取到}"
    fi
    echo
}

# 基线：worker=4（当前配置），后端默认
run_one "w4_default" train_dataloader.per_device_num_workers=4

# 加 worker
run_one "w8_default" train_dataloader.per_device_num_workers=8
run_one "w16_default" train_dataloader.per_device_num_workers=16

# 换视频解码后端为 torchcodec（按帧号精确解码，不做关键帧回退）
run_one "w8_torchcodec" \
    train_dataloader.per_device_num_workers=8 \
    train_dataloader.dataset.datasets.0.transforms.0.video_backend=torchcodec
run_one "w16_torchcodec" \
    train_dataloader.per_device_num_workers=16 \
    train_dataloader.dataset.datasets.0.transforms.0.video_backend=torchcodec

echo "=============== 扫描结束 ==============="
