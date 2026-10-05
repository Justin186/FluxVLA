#!/bin/bash
# 速度/显存扫描：找出单卡 24G 上最快的 (per_device_batch_size, grad_accum) 组合
# 每组跑 12 步，记录 s/it 与是否 OOM。共用同一个 work_dir 以复用统计缓存。

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG=configs/pi05/pi05_paligemma_tron2_cabinet_lora.py
WD=work_dirs/sweep
# 这个变量装的是 torchrun（名字沿用历史），自动探测，可用 FLUXVLA_TORCHRUN 覆盖
PY=${FLUXVLA_TORCHRUN:-$(dirname "$(command -v python3 2>/dev/null || echo /usr/bin/python3)")/torchrun}

rm -rf $WD
mkdir -p $WD

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for COMBO in "1 8" "2 4" "4 2"; do
    set -- $COMBO
    BS=$1
    GA=$2
    LOG=/tmp/sweep_bs${BS}_ga${GA}.log
    echo "=============== batch=$BS accum=$GA (有效 batch $((BS*GA))) gradient_ckpt=off ==============="
    timeout 1500 $PY --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
        --config $CFG --work-dir $WD \
        --cfg-options \
            train_dataloader.per_device_batch_size=$BS \
            runner.grad_accumulation_steps=$GA \
            runner.enable_gradient_checkpointing=False \
            runner.max_steps=12 \
        > $LOG 2>&1

    if tr '\r' '\n' < $LOG | grep -q "OutOfMemoryError"; then
        echo "  ❌ OOM"
        tr '\r' '\n' < $LOG | grep -m1 "OutOfMemoryError" | cut -c1-160 | sed 's/^/     /'
    else
        SIT=$(tr '\r' '\n' < $LOG | grep -oE "[0-9]+\.[0-9]+s/it" | tail -1)
        LOS=$(tr '\r' '\n' < $LOG | grep -oE "Loss :: [0-9.]+" | tail -1)
        echo "  ✅ 成功   ${SIT:-未取到}   ${LOS:-未取到}"
    fi
    echo
done
echo "=============== 扫描结束 ==============="
