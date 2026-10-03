# TRON2 训练机 · 双卡互联与训练吞吐实测

> 2026-10-02 在本机工作站（MSI PRO Z790-P WIFI + 2× RTX 4090）实测。
> 目的：把"双卡到底有没有用""重训要多久"从记忆/估算变成可复现的数字。

---

## 一页速览

| 问题 | 结论 |
|---|---|
| 这台机器能跑双卡吗 | **能。LoRA 配置下 1.84× 加速（92% 线性）** |
| 双卡会让单步更快吗 | **不会。**步时几乎不变（3.46 → 3.75 s），双卡是**每步吃 2 倍数据** |
| 那怎么省时间 | **必须同时把 `max_steps` 减半**，否则总时长基本不变 |
| GPU1 的 x1 链路有影响吗 | **LoRA 下几乎没有（8%）**，因为通信量极小；**全量微调下会致命** |
| 重训这个数据集要多久 | 双卡 17000 步 ≈ **18 小时**；双卡 8500 步 ≈ **9 小时** |

---

## 1. 硬件与互联拓扑

```
GPU0  0000:01:00.0   根端口 0000:00:01.0   max_width=16  max_speed=32 GT/s   ← CPU 直连
GPU1  0000:04:00.0   根端口 0000:00:1c.1   max_width=1   max_speed=8 GT/s    ← PCH，物理只有 1 条通道

拓扑        : GPU0 <-> GPU1 = PHB（走 PCIe 主机桥，非直连）
P2P 能力    : CNS（Chipset Not Supported）—— 不可用
torch 验证  : can_device_access_peer(0,1) = False，(1,0) = False
NVLink      : 无（nvidia-smi nvlink -s 无输出；RTX 4090 不支持）
```

**负载下实测链路**（空闲时会降频，必须负载下看）：

| | 负载下 |
|---|---|
| GPU0 | Gen4 × 16 |
| GPU1 | **Gen3 × 1** |

**跨卡拷贝带宽实测**（256 MB 张量，`b.copy_(a)`）：

| 方向 | 带宽 |
|---|---|
| GPU0 → GPU1 | **0.82 GB/s** |
| GPU1 → GPU0 | **0.82 GB/s** |

0.82 GB/s ≈ PCIe 3.0 × 1 的理论值，与链路状态吻合。

> **GPU1 的 `max_link_width=1` 是根端口自己的上限** —— 主板就是这么布线的，
> **换到别的插槽也一样**（其余槽位全部走 PCH，且都是 x1）。
> 想真正用上第二张卡，需要换支持 CPU 侧 x8+x8 拆分的主板，或换工作站/服务器平台。

---

## 2. 为什么 x1 链路没有拖垮 LoRA

关键在于**通信量**，不是链路宽度。

| 项目 | 量 | 在 0.82 GB/s 上的耗时 |
|---|---|---|
| 每样本 3 路图像上传（480×640 JPEG 解码后 → 224×224） | ~450 KB | 0.55 ms |
| **梯度同步**（LoRA，rank 32，只有适配器参数） | 数 MB 级 | **可忽略** |
| 对比：**全量微调**（3B 参数 × fp32） | **12 GB/步** | **≈ 15 秒（ring 约 29 秒）** |

**⇒ LoRA 下瓶颈不存在；全量微调下双卡会被这条链路彻底拖垮。**
（这也解释了"双卡比单卡还慢"的旧印象 —— 那是在全量微调口径下成立的。）

---

## 3. 实测吞吐

```bash
# 单卡
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  PATH=/home/lab/miniconda3/envs/fluxvla/bin:$PATH \
  bash scripts/train.sh configs/pi05/pi05_paligemma_tron2_cabinet_lora.py <WORK_DIR> \
  --cfg-options runner.max_steps=30

# 双卡
WANDB_MODE=disabled NPROC_PER_NODE=2 \
  PATH=/home/lab/miniconda3/envs/fluxvla/bin:$PATH \
  bash scripts/train.sh configs/pi05/pi05_paligemma_tron2_cabinet_lora.py <WORK_DIR> \
  --cfg-options runner.max_steps=30
```

| | 单卡 | 双卡 |
|---|---|---|
| 等效 batch | **24**（3×1×8） | **48**（3×2×8） |
| 稳态步时 | **3.46 s** | **3.75 s** |
| **吞吐** | **6.94 样本/秒** | **12.8 样本/秒** |
| **扩展效率** | — | **1.84×（92% 线性）** |
| 每步通信开销 | — | +0.29 s（≈8%） |
| GPU 利用率 | — | **99% / 100%** |
| 显存占用 | — | **20.4 / 20.8 GB**（共 24） |

> 第 1 步是预热（双卡 12.0 s），之后稳定。另跑了一次 **200 步长测**确认稳态：
> 200 步 13:24，稳态 3.78~3.82 s/步，Loss 0.39 → 0.086。

**两条卡都跑到 99~100%** —— 说明训练是**纯 GPU-bound**，x1 链路在计算期间没有参与。

---

## 4. 检查点落盘开销（实测）

第 150 步触发的落盘：

| 文件 | 大小 |
|---|---|
| `step-000150-...pt` | 14 GB |
| `step-000150-...safetensors` | 14 GB |
| `latest-checkpoint.*` | 指针（几十字节） |

```
单次落盘      ≈ 28 GB / 40 秒
磁盘占用变化  : 268 GB → 240 GB 可用
max_keep_ckpts=3  峰值 (3+1) × 28 = 112 GB
```

> **`.pt` 与 `.safetensors` 内容完全重复。** 只保留一种格式：
> 单次落盘 **14 GB / 20 秒**，峰值 **56 GB**。
> 按 17 次落盘算，省 **~6 分钟**和 **56 GB** 磁盘 —— 磁盘只剩 240 GB 时这条更值钱。

---

## 5. 训练时间估算

以双卡稳态 **3.8 s/步**、启动 ~2 分钟、每次落盘 40 秒计：

| 方案 | 步数 | 算力 | 落盘 | 启动 | **总计** |
|---|---|---|---|---|---|
| 双卡，沿用 `max_steps=17000` | 17000 | 17.9 h | 17 次 ≈ 11 min | 2 min | **≈ 18.2 h** |
| **双卡，`max_steps=8500`** | 8500 | 9.0 h | 8 次 ≈ 5 min | 2 min | **≈ 9.1 h** |
| 单卡，`max_steps=17000` | 17000 | 16.3 h | 17 次 ≈ 11 min | 2 min | ≈ 16.7 h |

> **加卡不省时间。** 17000 步在双卡下意味着把数据集过两倍的遍数；
> 要拿到同样的训练量，必须把步数减半 —— 而且**必须同步改 `decay_steps`**：

```python
lr_scheduler = dict(..., warmup_steps=1000, decay_steps=17000)   # 与 max_steps 绑定
```

**只改 `max_steps` 不改 `decay_steps`，余弦退火只走一半**，模型会停在偏高的学习率上。
另外等效 batch 从 24 变 48，**学习率通常需要同步调整**（线性缩放即 ×2，有 warmup 时可保守些）。

---

## 6. 其它发现

- **数据加载不是瓶颈**：3 路视频解码（torchcodec）稳态 **0.004 s/样本**，最坏 0.008 s，
  对比每样本 ~0.16 s 的算力 —— 只占 **2~4%**。加 dataloader worker 不会更快。
- **存储不拖后腿**：数据集 2.4 GB，NVMe 挂 `0000:00:06.0`（CPU 直连 Gen4 ×4），
  **不和 GPU1 抢 DMI**。
- **显存是 batch 的真正上限**：20.8 GB / 24 GB，所以 `per_device_batch_size=4` 会 OOM。
  已开的优化：`enable_gradient_checkpointing=True`、`mixed_precision_dtype='bf16'`、
  `grad_accumulation_steps=8` —— 无明显漏项。
- **跑训练前必须设 `WANDB_MODE=disabled`**（如果不用 wandb）。
  `base_train_runner.py:192` 用 `os.environ.get('WANDB_MODE', 'online')` 取值且默认 `online`，
  而 `ddp_train_runner.py:930` 在 `wandb_mode != 'disabled'` 时直接调 `wandb.log()`，
  **没有 `wandb.init()`，会在第 0 步崩掉**。

---

## 7. 复现命令

```bash
# 1) 确认没有别的进程占卡
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 2) 链路与 P2P（空闲时 gen/width 会降频，必须负载下再确认一次）
nvidia-smi topo -m
nvidia-smi topo -p2p r
cat /sys/bus/pci/devices/0000:01:00.0/current_link_width   # GPU0
cat /sys/bus/pci/devices/0000:04:00.0/current_link_width   # GPU1

# 3) 跨卡带宽
python - <<'PY'
import torch, time
N = 256*1024*1024//4
a = torch.ones(N, dtype=torch.float32, device='cuda:0')
b = torch.zeros(N, dtype=torch.float32, device='cuda:1')
def bw(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return N*4*n/(time.perf_counter()-t)/1e9
print('GPU0->GPU1 %.2f GB/s' % bw(lambda: b.copy_(a)))
print('GPU1->GPU0 %.2f GB/s' % bw(lambda: a.copy_(b)))
PY
```

---

## 附：主板槽位全貌（MSI PRO Z790-P WIFI / MS-7E06，BIOS A.90）

```
0000:00:01.0   max_width=16   max_speed=32 GT/s   ← CPU 直连（主显卡槽）
0000:00:06.0   max_width=4    max_speed=16 GT/s   ← CPU 直连（NVMe 在用）
0000:00:1c.0   max_width=1    max_speed=8 GT/s    ← PCH x1
0000:00:1c.1   max_width=1    max_speed=8 GT/s    ← PCH x1（GPU1 在这）
0000:00:1c.2   max_width=1    max_speed=5 GT/s    ← PCH x1
```

**只有一条快通道。** 第二个 x16 槽电气上是 PCH x1 —— 所以"把卡换到另一个槽"无效。
