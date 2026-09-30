# TRON2 柜内抓取 · PI0.5 LoRA 交接文档

> **这份文档的目标**：在一台**只有 `tron_ws` 目录的 Ubuntu 20 空白机器**上，
> 从零把训练 + 推理环境搭起来，复现已训好的模型，并接到机器人上。
>
> 所有版本号、路径、命令都是**本机实测**的，不是推断。凡是有"实测"标记的，
> 都是真跑过并拿到了结果的。

---

## 0. 现状速览

| 项目 | 状态 |
|---|---|
| **训练** | ✅ 已完成。17000 步 / 20 小时 27 分 / 最终 loss **0.0088**（比开训降 9.2 倍，全程 0 次尖峰） |
| **代码** | ✅ 已在 git：你的 fork `github.com/Justin186/FluxVLA`，分支 `main`，6 个提交 |
| **权重 / 数据** | ⚠️ **不在 git 里**（被 gitignore），需要 scp。**目前世界上只有原始机器那一份** |
| **部署代码改动** | ✅ 已落地并测试（41 个单测通过） |
| **上机器人** | ❌ **未做**。见 §9 待办 |

### 首要提醒

**训练产物（模型 + 数据集）现在只存在于一台临时机器上。** 在把环境搭好之前，
先把 §3.3 的 scp 做掉，否则 20 小时的计算成果会随机器释放一起消失。

---

## 1. 目标机要求

### 1.1 硬件

| | 要求 | 实测环境 |
|---|---|---|
| GPU | **单张 ≥ 24 GB 显存** | RTX 4090 D，24564 MiB（计算能力 sm_89） |
| CPU | 建议 ≥ 16 核（数据加载瓶颈） | 44 核 |
| 内存 | ≥ 32 GB | 78 GB |
| 磁盘 | **≥ 200 GB 可用** | 见下方说明 |

**显存为什么必须 24 GB**：micro-batch 4 会 OOM，3 是上限（实测峰值 20538 MiB）。
16 GB 的卡跑不了这个配置，需要下调 batch（见 §5.4）。

**磁盘为什么必须 200 GB**：单个检查点 **29.2 GB**（`.pt` 14.75 GB + `.safetensors` 14.47 GB，
两者内容重复），保留 3 份就是 87.6 GB，加上保存时"先写新的再删旧的"峰值要 **116.8 GB**。
再加上基座权重 6.8 GB。

### 1.2 系统

| | 实测环境 | 你的机器 | 备注 |
|---|---|---|---|
| OS | **Ubuntu 22.04.5 LTS** | **Ubuntu 20.04** | ⚠️ **有差异，见 §1.4** |
| 内核 | 5.4.250（velinux 定制） | — | |
| NVIDIA 驱动 | **535.154.05** | 建议 ≥ 535 | `nvidia-smi` 报 CUDA **12.2** |
| CUDA Toolkit | 未装（用 wheel 内自带的） | 可不装 | 但 `install_env.sh` 会探测 nvcc 决定 profile |

### 1.3 软件基线

| | 版本 |
|---|---|
| conda | 26.7.1，装在 `/opt/miniconda3` |
| conda env 名 | `fluxvla` |
| Python | 3.10.21 |
| PyTorch | **2.6.0+cu124** |

> **`fluxvla` 这个 env 名字不要改** —— 仓库里多处脚本硬编码了
> `/opt/miniconda3/envs/fluxvla/bin/python`。如果装在别的位置，要么改脚本，
> 要么建个同名软链。

### 1.4 ⚠️ Ubuntu 20 vs 22 的差异

实测环境是 22.04，你们是 20.04。需要注意的点：

1. **GLIBC 版本**：20.04 是 glibc 2.31，22.04 是 2.35。绝大多数 torch wheel 都能跑，
   但**如果 pip 装到的是为 glibc 2.35 编译的 wheel，import 时会报 `GLIBC_2.34 not found`**。
   遇到就指定较老的 wheel（`torch==2.6.0+cu124` 官方 wheel 是兼容 2.31 的）。
2. **Python 3.10 需要自己装 conda**，20.04 系统自带是 3.8，不要用系统的。
3. **gcc**：实测环境是 gcc 8.3。20.04 默认 gcc 9。如果后续要编译 CUDA 扩展，
   `requirements` 里的 flash-attn 是**预编译 wheel**，正常情况不需要本地编译。
4. **`sysctl` / 共享内存**：如果 `/dev/shm` 小于 8 GB，DataLoader 多 worker 会报
   `bus error`。用 `df -h /dev/shm` 检查，不够就加 `--shm-size`（容器）或改 sysctl。

---

## 2. 目录布局（关键，不要改）

**所有路径都按这个结构硬编码在配置里。** 换位置就要同时改配置。

```
tron_ws/
├── FluxVLA/                              ← git clone 你的 fork
│   ├── checkpoints/
│   │   └── pi05_base_bf16/               ← scp 来的基座权重（6.8 GB）
│   ├── datasets/
│   │   └── RealRobot_Tron2_lerobot/      ← scp 来的数据集（实占 2.4 GB）
│   │       ├── lerobot_2026-09-28_22-05-37/
│   │       ├── lerobot_2026-09-28_22-26-43/
│   │       ├── lerobot_2026-09-28_22-36-10/
│   │       ├── lerobot_2026-09-28_22-44-32/
│   │       ├── lerobot_2026-09-28_22-54-05/
│   │       └── tron2_stats_armsymmetric.json      ← 已在 git 里
│   └── work_dirs/
│       └── tron2_cabinet_lora_v2/        ← scp 来的训练产物
│           ├── dataset_statistics.json   ← ★ 必须在 checkpoints 的上两级
│           └── checkpoints/
│               └── step-017000-epoch-007-loss=0.0088.safetensors   ← 13.5 GB
└── datasets_raw/                         ← ⚠️ 见 §3.3.3，videos 软链的目标
```

### 为什么 `dataset_statistics.json` 位置不能动

推理代码里有一句硬断言（`fluxvla/engines/runners/base_inference_runner.py`）：

```python
data_stat_path = os.path.join(Path(ckpt_path).resolve().parent.parent,
                              'dataset_statistics.json')
assert os.path.exists(data_stat_path), \
    f'Dataset statistics file not found at {data_stat_path}!'
```

即 **`ckpt_path` 的上两级目录**里必须有它。层级错了服务起不来。

---

## 3. 从零搭建

### 3.1 装 conda 与环境

```bash
# 1) 装 miniconda 到 /opt/miniconda3（和实测环境一致，省得改脚本）
#    下载对应架构的安装包后：
sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
export PATH=/opt/miniconda3/bin:$PATH
source /opt/miniconda3/etc/profile.d/conda.sh

# 2) 建环境（Python 3.10，名字必须是 fluxvla）
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
```

### 3.2 clone 代码

```bash
mkdir -p ~/tron_ws && cd ~/tron_ws
git clone https://github.com/Justin186/FluxVLA.git
cd FluxVLA

# 确认在这个提交上（6 个业务提交都在 main 上）
git log --oneline -6
# e0eb092 [Fix] Expand TRON2 policy actions to the robot's 18-dim layout
# f51b41e [Feat] Add accelerated deployment config for TRON2 cabinet policy
# d567b98 [Chore] Relax CUDA profile check for local driver
# b7a0115 [Feat] Add TRON2 cabinet LoRA training config, corrected stats and docs
# 226df18 [Fix] Support ParquetDatasetV3 in auto statistics and terminal padding
# be32110 [Docs] Announce technical report release in multilingual READMEs
```

### 3.3 拉数据与权重（scp）

> **以下四份都不在 git 里**（`checkpoints/*`、`datasets/*`、`work_dirs/` 都在 `.gitignore` 内）。

#### 3.3.1 权重（必需，共 20.3 GB）

```bash
# 训练好的模型 —— 合并后的完整模型，推理只需这一个就够
scp "root@<源机器>:/root/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/checkpoints/step-017000-epoch-007-loss=0.0088.safetensors" \
    ~/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/checkpoints/

# 推理硬断言需要的小文件（5.9 KB）
scp root@<源机器>:/root/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/dataset_statistics.json \
    ~/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/

# 基座权重（6.8 GB）—— 只有重新训练才需要；纯推理可省
scp -r root@<源机器>:/root/tron_ws/FluxVLA/checkpoints/pi05_base_bf16 \
    ~/tron_ws/FluxVLA/checkpoints/
```

> 文件名里有 `=`，**scp 时务必加引号**，否则某些 shell 会解析出错。

#### 3.3.2 数据集（2.4 GB）

```bash
scp -r root@<源机器>:/root/tron_ws/FluxVLA/datasets/RealRobot_Tron2_lerobot \
    ~/tron_ws/FluxVLA/datasets/
```

#### 3.3.3 ⚠️ 数据集里藏着软链陷阱

源机器上每个子集的 `videos/` 目录**不是真目录，是指向绝对路径的软链**：

```
datasets/RealRobot_Tron2_lerobot/lerobot_2026-09-28_22-05-37/videos
  -> /root/tron_ws/datasets_raw/lerobot_2026-09-28_22-05-37/videos
```

后果：
- `du -sh` 只报 **18 MB**，实际是 **2.4 GB**（软链不被计入）
- **直接 `scp -r` 会把软链原样拷过去，到新机器上是断链** → 训练时读不到视频

**正确做法（三选一）**：

```bash
# 方案 A：不解引用，把 datasets_raw 一起搬，并在新机器上保持同样的绝对路径
scp -r root@<源机器>:/root/tron_ws/datasets_raw ~/tron_ws/
#   要求新机器的路径也是 /root/tron_ws/datasets_raw，否则软链还是断的

# 方案 B：打包时解引用（推荐）
tar -czhf tron2_data.tar.gz -C ~/tron_ws/FluxVLA datasets/RealRobot_Tron2_lerobot
#        ↑ h = 跟随软链，把真实视频文件打进去
# 拷过去后解包即可，得到的是真目录

# 方案 C：rsync 解引用
rsync -avL root@<源机器>:/root/tron_ws/FluxVLA/datasets/RealRobot_Tron2_lerobot/ \
      ~/tron_ws/FluxVLA/datasets/RealRobot_Tron2_lerobot/
```

**验证有没有拷对**：

```bash
cd ~/tron_ws/FluxVLA
du -sh datasets/RealRobot_Tron2_lerobot
# 期望 ~2.4 GB。只报十几 MB 就是软链没解开

find datasets/RealRobot_Tron2_lerobot -type l | wc -l
# 期望 0。非 0 说明还有软链

find datasets/RealRobot_Tron2_lerobot -name '*.mp4' | wc -l
# 期望有值（每个子集每个相机一个 mp4）
```

### 3.4 装依赖

```bash
cd ~/tron_ws/FluxVLA
conda activate fluxvla

# 关键：real-only 模式 = 训练 + 真实机器人 / 远程推理依赖
bash scripts/install_env.sh real-only --profile cu124
```

**为什么必须 `--profile cu124`**：实测环境 `nvidia-smi` 报 CUDA **12.2**，
按 `auto` 探测会选错 profile。cu124 的 wheel 在这个驱动上实测可用。

**⚠️ `install_env.sh` 里有一处本人打的本地补丁**（第 544 行，`git log` 里的
`[Chore] Relax CUDA profile check for local driver`）：它把"驱动 CUDA 版本不满足
profile 要求"这个检查**从 `exit 1` 改成只打警告**。

```bash
# 补丁长这样（scripts/install_env.sh:544）
echo "  [本地补丁] 忽略该检查：torch cu124 已实测可在本机驱动（535/CUDA 12.2 报告版本）上正常使用 GPU。" >&2
```

这是为了解决"驱动报 12.2 但要用 cu124 wheel"的矛盾。**如果你们机器的驱动
正常支持 cu124，这条补丁其实不需要** —— 但它不会造成危害（只是少一次报错退出）。
**建议不要把这个提交往官方提 PR。**

**装完必须验证 torch 真能用 GPU**（这一步能提前发现 §1.4 的 glibc 问题）：

```bash
/opt/miniconda3/envs/fluxvla/bin/python -c "
import torch
print('torch      :', torch.__version__)
print('cuda 运行时:', torch.version.cuda)
print('GPU        :', torch.cuda.get_device_name(0))
print('compute cap:', torch.cuda.get_device_capability(0))
print('bf16 支持  :', torch.cuda.is_bf16_supported())
x = torch.randn(1000, 1000, device='cuda')
print('矩阵乘验证 :', (x @ x).sum().item() != 0)
"
```

期望输出（实测）：

```
torch      : 2.6.0+cu124
cuda 运行时: 12.4
GPU        : NVIDIA GeForce RTX 4090 D
compute cap: (8, 9)
bf16 支持  : True
```

### 3.5 装测试依赖（可选，但推荐）

```bash
/opt/miniconda3/envs/fluxvla/bin/python -m pip install pytest

# 跑我们新增的布局测试，确认代码没问题
cd ~/tron_ws/FluxVLA
/opt/miniconda3/envs/fluxvla/bin/python -m pytest test/test_transforms/ test/test_datasets/ -q
# 期望：41 passed
```

---

## 4. 环境版本清单（实测）

搭完环境后用这个脚本核对，任何一项差太多都要留意。

```bash
/opt/miniconda3/envs/fluxvla/bin/python -c "
import importlib
for m in ['torch','torchvision','transformers','numpy','mmengine','accelerate',
          'diffusers','timm','einops','peft','safetensors','pyarrow','pandas',
          'av','torchcodec','imageio','cv2','triton','flash_attn','wandb','zmq',
          'msgpack','PIL']:
    try:
        mod = importlib.import_module(m)
        print('%-16s %s' % (m, getattr(mod, '__version__', '?')))
    except Exception:
        print('%-16s 未安装' % m)
"
```

| 包 | 实测版本 | 说明 |
|---|---|---|
| torch | **2.6.0+cu124** | |
| torchvision | 0.21.0+cu124 | |
| transformers | 5.3.0 | |
| numpy | **1.26.4** | ⚠️ 必须 1.x，2.x 会和 mmengine/opencv 冲突 |
| mmengine | 0.10.7 | 配置系统 |
| accelerate | 0.33.0 | |
| diffusers | 0.37.0.dev0 | 开发版，正常 |
| timm | 0.9.10 | |
| einops | 0.4.1 | |
| **peft** | **0.19.1** | LoRA 必需 |
| safetensors | 0.8.0 | 检查点格式 |
| pyarrow | 24.0.0 | parquet 读取 |
| pandas | 2.3.3 | |
| **av** | **14.2.0** | 视频解码 |
| **torchcodec** | **0.2.1** | ⚠️ **必须和 torch 版本配对**：torch 2.6 → 0.2.1，torch 2.8 → 0.7.0 |
| imageio | 2.37.3 | |
| cv2 (opencv) | 4.11.0 | |
| **triton** | **3.2.0** | 加速推理必需 |
| **flash_attn** | **2.8.3** | 预编译 wheel |
| wandb | 0.21.0 | 装了但**不要开**，见 §7.5 |
| zmq (pyzmq) | 27.1.0 | 远程推理 |
| msgpack | 1.2.1 | 远程推理 |
| PIL (Pillow) | 12.3.0 | |

**视频解码后端**（`fluxvla/datasets/utils/video_decode.py`）：优先用 **torchcodec**，
失败回退到 torchvision。torchcodec 版本对不上会直接抛
`matching PyTorch (torch 2.8: torchcodec 0.7.0; torch 2.6: torchcodec 0.2.1)`。

---

## 5. 训练

### 5.1 启动命令

```bash
cd ~/tron_ws/FluxVLA
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

/opt/miniconda3/envs/fluxvla/bin/torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora.py \
  --work-dir work_dirs/tron2_cabinet_lora_v2
```

> **⚠️ 必须用 `torchrun`，不能 `python scripts/train.py`。**
> 直跑会在初始化阶段就崩：
> ```
> fsdp_train_runner.py:117  device_id = overwatch.local_rank()
> AttributeError: 'PureOverwatch' object has no attribute 'local_rank'
> ```
> 原因：`overwatch` 的分布式上下文没初始化。`torchrun` 会注入 `RANK`/`LOCAL_RANK`。

### 5.2 已标定的超参数

配置文件 `configs/pi05/pi05_paligemma_tron2_cabinet_lora.py`：

| 参数 | 值 | 说明 |
|---|---|---|
| `per_device_batch_size` | **3** | 4090 24G 上限（4 会 OOM） |
| `grad_accumulation_steps` | **8** | 有效 batch = 3 × 1 × 8 = **24** |
| `max_steps` | **17000** | 56485 帧 × 7.2 epoch ÷ 24 |
| `lr` | **2.5e-5** | LoRA 尺度，不是全量微调的尺度 |
| `warmup_steps` | 1000 | |
| `decay_steps` | **17000** | 与 max_steps 相等 = 完整余弦退火到 `min_lr` |
| `min_lr` | 2.5e-6 | |
| `save_iter_interval` | **1000** | ≈20 分钟一次，覆盖式滚动保存 |
| `max_keep_ckpts` | **3** | 见 §7.6 的磁盘约束 |
| `use_lora` / `lora_rank` | True / 32 | |
| `loss_action_dim` | 16 | 真实动作维度 |
| `max_action_dim` | 32 | 补齐后的维度 |
| `enable_gradient_checkpointing` | **True** | 实测 ON/OFF 显存与速度完全相同，开着当保险 |
| `active_trackers` | **`('jsonl',)`** | 见 §7.5 |

**实测性能**：稳态 **4.21 s/步** → 17000 步 ≈ **20 小时**，峰值显存 **20538 MiB**。

### 5.3 训完的验证标准

```bash
/opt/miniconda3/envs/fluxvla/bin/python -c "
import json, statistics, glob
f = sorted(glob.glob('work_dirs/tron2_cabinet_lora_v2/pi05_paligemma_tron2_cabinet_lora_*.jsonl'))
ls = [json.loads(l) for l in open(f[-1]) if l.strip()]
loss = [x['VLA Train/Loss'] for x in ls]
print('步数         :', len(ls), '(期望 17000)')
print('全程中位数   : %.4f (期望 ~0.011，含早期高 loss)' % statistics.median(loss))
print('末段1000步   : %.4f (期望 ~0.009)' % statistics.median(loss[-1000:]))
print('最后 lr      :', ls[-1]['VLA Train/Learning Rate'], '(期望 2.5e-06)')
print('尖峰(loss>1) :', sum(1 for l in loss if l > 1), '次 (期望 0)')
"
```

**我们的结果**：17000 步 / 全程中位 0.0113 / 末段 1000 步 **0.0081** /
最后 lr 2.5e-06 / 尖峰 0 次。

> 判读要点：**看"末段"而不是"全程"**。全程中位数被前 1000 步的高 loss
> （开训时 0.08 左右）拉高了，不能反映收敛水平。

### 5.4 换机器要调的参数

显存不是 24 GB 时，按这个改（**有效 batch 想保持 24**）：

| 显存 | `per_device_batch_size` | `grad_accumulation_steps` | `max_steps` | 备注 |
|---|---|---|---|---|
| ≥ 24 GB | 3 | 8 | 17000 | 当前配置 |
| 16–24 GB | 2 | 12 | 17000 | 显存余量更大但慢 ~18% |
| < 16 GB | 1 | 24 | 17000 | 很慢，考虑开 CPU offload |

**注意**：`max_steps` 必须能被 `save_iter_interval` 整除，否则训练循环结束时
**没有收尾保存**，最后一步不落盘（见 §7.6）。

---

## 6. 推理

### 6.1 部署用的配置文件

已经在 git 里：`configs/pi05/pi05_paligemma_tron2_cabinet_lora_deploy.py`

它用 mmengine 的 `_base_` 复用训练配置，**只覆盖 `inference_model`**，换成
Triton + CUDA Graph 加速版。

### 6.2 启动服务

```bash
cd ~/tron_ws/FluxVLA
export TOKENIZERS_PARALLELISM=false

scripts/zmq_inference_server.sh \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora_deploy.py \
  --ckpt-path work_dirs/tron2_cabinet_lora_v2/checkpoints/step-017000-epoch-007-loss=0.0088.safetensors \
  --port 5555
```

**首次调用会花 29.3 秒**（Triton JIT 编译 + CUDA Graph 捕获），日志里会打：

```
[Triton Inference] Recording CUDA Graph ...
[Triton Inference] CUDA Graph recorded successfully!
```

**⚠️ 服务起来后必须先 warmup 一次再让机器人动** —— 否则第一次推理会有 29 秒卡顿。

### 6.3 实测延迟

测法：加载 `step-017000-*.safetensors`（`use_lora=False`，0 missing / 0 unexpected），
`torch.autocast('cuda', dtype=torch.bfloat16)`，warmup 后计时 20 次取中位数。

| 路径 | 稳态延迟 | 占预算 | 加速比 |
|---|---|---|---|
| 朴素 `PI05FlowMatching` | 236.3 ms（230.6 ~ 237.7） | 22.1% | 1× |
| **加速 `PI05FlowMatchingRTCInference`** | **45.8 ms**（45.7 ~ 45.9） | **4.3%** | **5.16×** |

**延迟预算不是 33 ms，是 1067 ms** —— 一次推理输出 32 步动作，
机器人按 30 Hz 执行：`32 ÷ 30 = 1.067 秒`。

**结论：延迟完全不是瓶颈**，连朴素路径都只占 22%。

> ⚠️ **加速比是 5.16×，不是 `docs/inference_acceleration.md` 里写的 15×。**
> 那个 15× 是在 **A100** 上标的，sm_89（4090）实测只有 5.16×。

### 6.4 推理接口契约（实测，和 docstring 不符）

```python
# 朴素版 PI05FlowMatching
images      : (B, num_views * 3, H, W)     # ★ 展平的，不是 (B, V, 3, H, W)
img_masks   : (B, num_views)  bool
lang_tokens : (B, L)  int64
lang_masks  : (B, L)  bool
states      : (B, 32)                      # 16 维真实 + 16 维补齐
# 输出      : (1, 50, 32)  float32         # n_action_steps=50, max_action_dim=32
```

**`images` 的形状是最容易踩的**：docstring 写 `(bsize, 3, 224, 224)`，
实测必须是 `(B, V*3, H, W)`。传 `(B, V, 3, H, W)` 会报
`ValueError: too many values to unpack (expected 4)`。

### 6.5 ⚠️ 必须用 RTC 版加速类

```
PI05FlowMatchingInference        ✗ RuntimeError: tensor a (2) vs b (3)   ← 3 路相机报错
PI05FlowMatchingRTCInference     ✓ 正常，45.8 ms
```

**`PI05FlowMatchingInference`（非 RTC）在 3 路相机下会崩。**
`docs/inference_acceleration.md` 里的 π0.5 范例写的是非 RTC 版，
**直接照抄会在我们的 3 相机配置上失败**。部署配置里已经用的是 RTC 版。

---

## 7. 必须知道的坑

### 7.1 动作布局：模型 16 维 vs 机器人 18 维

| | 布局 | 夹爪位置 | 维度 |
|---|---|---|---|
| **模型输出** | `[左臂7, 左夹爪, 右臂7, 右夹爪]` | 7, 15 | **16** |
| **机器人期望** | `[左臂7, 右臂7, 头部2, 左夹爪, 右夹爪]` | 16, 17 | **18** |

**已于本仓库修复**：新增 `fluxvla/transforms/normalize.py::DenormalizeTron2Action`，
在反归一化时做展开。放在这个位置而不是 runner 里，是因为
**本机直跑和 ZMQ 服务两条路径都会调用 `denormalize_action`**，一处覆盖两条。

配置侧对应：

```python
denormalize_action=dict(
    type='DenormalizeTron2Action',     # ← 原来是 DenormalizeDeltaAction
    norm_type='quantile',
    action_dim=16,
    delta_action_mask=[True]*7 + [False] + [True]*7 + [False],
    state_permutation=[0,1,2,3,4,5,6, 16, 7,8,9,10,11,12,13, 17, 14,15],
)
```

**`state_permutation` 的长度必须是 18，不是 16。** `DenormalizeDeltaAction`
只能重排、不能筛选（校验是 `must contain every index in [0, D) exactly once`），
而机器人原始状态是 18 维。写成 16 直接抛错：

```
ValueError: state_permutation must contain every index in [0, D) exactly once.
```

正确写法把模型要用的 16 位放**前 16 位**（`delta_action_mask` 只消费
`state[:16]`），头部两位放末尾（不参与）。**不设或设错的后果（实测）**：
dim 8–14 会吃到机器人状态索引 8..14 = 右臂[1..6] + `head_pitch`，
**右臂整体错位一格并混入头部值**。

**验证**：`test/test_transforms/test_tron2_action_layout.py`（11 个用例）。
其中两条是防回退的：长度 16 的 permutation 必须报错、
16 维动作必须让 runner 的 `actions[:, 16]` 抛 `IndexError`。

### 7.2 左臂 + 左夹爪是退化维度

**要按"这两个通道不可用"来设计部署。**

| 现象 | 证据 |
|---|---|
| 左臂全程静止 | 全部 56888 帧左臂关节恒定 |
| **左夹爪全程恒为 0** | 统计量 `q01=0 q99=0 min=0 max=0 std=0` |

后果：
- 模型对左臂/左夹爪**没有有效学习信号**
- 左夹爪量化区间退化 → **反归一化把任何输入都塌成 0**，输出恒为 0

**部署建议**：对左臂输出做安全钳制、或直接锁定左臂；
**若工艺需要左夹爪动作，必须另行实现，不要指望模型输出。**

### 7.3 stats 必须用修正版

**必须用 `datasets/RealRobot_Tron2_lerobot/tron2_stats_armsymmetric.json`**
（已提交到 git），不能用自动统计的版本。

原因：自动统计下左臂维度会按 **约 150 倍**的尺度反归一化 → **左臂指令直接飙飞**。
这个文件是**手工修正**过的产物，无法自动重生成 —— 这也是它被强制加进 git
（原本在 `datasets/*` 的 gitignore 里）的原因。

### 7.4 检查点保存极其昂贵

LoRA 保存**不是**"存 290 MB 适配器"，而是
**重建基座 → 重新加载 7.23 GB 权重 → 合并 LoRA → 写出完整模型**，而且
**同时写 `.pt` 和 `.safetensors` 两份**（内容重复）：

| | 大小 |
|---|---|
| 单份 `.pt` | 14.75 GB |
| 单份 `.safetensors` | 14.47 GB |
| **单份合计** | **29.2 GB** |
| **单次保存耗时** | **30 ~ 110 秒** |

### 7.5 wandb 会挂起，只用 jsonl

`VLAMetric` 的 wandb tracker 里 `wandb.init()` **没有 try/except**：

```python
# fluxvla/engines/metrics/vla_metric.py
wandb.init(...)      # ← headless 且无 WANDB_API_KEY 时挂起或报错
```

而且 `WandBTracker.finalize()` 里有个 **`time.sleep(210)`**。

所以配置里设的是：

```python
metric=dict(type='VLAMetric', active_trackers=('jsonl',), ...)
```

**如果你之后配好了 wandb 登录，可以改回 `('jsonl', 'wandb')`。**
另外跑之前建议 `export WANDB_MODE=disabled` 兜底。

### 7.6 保存间隔与 max_steps 的整除约束

```python
# base_train_runner.py:570
def _should_save_step_checkpoint(self):
    return (self.metric.global_step % self.save_iter_interval) == 0
```

**步骤态下只有 `save_iter_interval` 这一条保存路径，而且训练循环结束时
没有收尾保存。** 所以：

> **`max_steps` 必须能被 `save_iter_interval` 整除，否则最后一步不落盘。**

改这两个值时要一起算。当前是 `17000 % 1000 == 0` ✅。

**磁盘峰值约束**（保存是"先写新的、再删旧的"，峰值多占一份）：

| `max_keep_ckpts` | 稳态 | 峰值 | 200 GB 盘上的余量 |
|---|---|---|---|
| 2 | 58.4 GB | 87.6 GB | 充足 |
| **3（当前）** | **87.6 GB** | **116.8 GB** | **~83 GB** |
| 4 | 116.8 GB | 146.0 GB | 很紧张 |

### 7.7 环境残留可能有 PYTHONPATH

如果目标机之前装过东西，**跑训练前 `unset PYTHONPATH`**。

我们踩过这个坑：残留的 `PYTHONPATH` 让 Python 导入了系统的 `send2trash`，
导致**每次删检查点都进回收站而不是真删**，磁盘悄悄涨了 28 GB。
清理：

```bash
/bin/rm -rf ~/.local/share/Trash/files/*
df -h /
```

### 7.8 首次推理 29 秒

见 §6.2。**服务起来后必须先 warmup。**

---

## 8. 故障排查速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `'PureOverwatch' object has no attribute 'local_rank'` | 用 `python` 直跑 | 改用 `torchrun`（§5.1） |
| `GLIBC_2.34 not found` | wheel 为更新 glibc 编译 | 换较老的 wheel（§1.4） |
| `CUDA out of memory`（训练） | batch 太大 | 降到 2 或 1（§5.4） |
| `CUDA out of memory`（保存时） | **有别的进程占着显存** | `pkill -f scripts/train.py`，确认 `nvidia-smi` 干净再启 |
| `dataset_statistics file not found at ...` | 层级不对 | 放到 `ckpt_path/../..`（§2） |
| `state_permutation must contain every index` | permutation 写成 16 长度 | 改成 18（§7.1） |
| `IndexError` 在 `actions[:, 16]` | transform 没换成 `DenormalizeTron2Action` | 见 §7.1 |
| `tensor a (2) vs b (3)` | 用了非 RTC 加速类 | 换 `PI05FlowMatchingRTCInference`（§6.5） |
| `too many values to unpack (expected 4)` | `images` 形状不对 | 用 `(B, V*3, H, W)`（§6.4） |
| `matching PyTorch (torch 2.8: torchcodec 0.7.0 ...)` | torchcodec 版本不配对 | torch 2.6 → torchcodec 0.2.1 |
| 训练卡住不动、日志无输出 | wandb 在等登录 | `export WANDB_MODE=disabled`（§7.5） |
| 磁盘莫名增长 | 删的东西进了回收站 | `unset PYTHONPATH` + 清回收站（§7.7） |
| 视频读不到 / 帧全黑 | 数据集软链没解开 | §3.3.3 |

### 有用的监控命令

```bash
# 训练进度（每次看一行就是最新状态）
tail -1 work_dirs/tron2_cabinet_lora_v2/pi05_paligemma_tron2_cabinet_lora_*.jsonl

# GPU
watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'

# 检查点
ls -lh work_dirs/tron2_cabinet_lora_v2/checkpoints/

# 磁盘
df -h /
```

---

## 9. 未完成 / 待办

### 9.1 立即要做的

1. **把权重和数据从原机器搬出来**（§3.3）。它们现在还只存在于一台临时机器上。
2. 按 §3 搭好环境，跑通 §3.5 的 41 个测试。

### 9.2 上机器人之前

3. **真实观测跑一次推理**，检查输出的 18 维动作数值是否合理
   （重点看：左右臂关节是否在量程内、右夹爪是否符合预期、
   **左夹爪会恒为 0 属预期**，见 §7.2）。
4. **左臂安全钳制**。左臂是退化维度，必须先做限幅或锁定，再让它接机器人。
5. **腕部相机首帧检查**。采集时腕部相机有 1~6 帧（33~200 ms）启动延迟，
   首帧可能是黑的。建议加"首帧有效再推理"的检查。
6. **`prepare_pose` 确认**。单机器人的初始位姿和采集时是否一致。

### 9.3 还没验证过的

7. **真实数据上的推理质量**。目前只验证了延迟和形状，**没有验证过
   实际抓取成功率** —— 因为手上没有完整的评测环境。
8. **ZMQ 远程路径的端到端**。服务端和 `denormalize_action` 逻辑都验证了，
   但机器人侧客户端没跑过真实链路。
9. **Orin 板端部署**。`docs/orin_docker_runtime.md` / `docs/orin_flashing.md`
   有官方文档，但没在 Orin 上验证过这个模型。

---

## 附录 A：这个环境里跑过什么

留个记录，方便理解哪些数字是可复现的。

| 操作 | 结果 |
|---|---|
| batch / 显存扫描 | batch 2 → 16480 MiB；batch 3 → 20538 MiB；batch 4 → OOM |
| 梯度检查点 ON/OFF 对比 | 都是 1.75 s/it、16480 MiB → 无差别，保持 ON |
| 冒烟测试（18 步） | 通过，无线程池错误、无 OOM |
| 检查点保存测试 | 成功产出 2 份，单份 29.2 GB；中途 OOM 系另一进程抢占显存所致 |
| **正式训练** | **17000 步 / 20h27m / loss 0.212 → 0.0088 / 0 次尖峰** |
| 推理延迟 | 朴素 236.3 ms / 加速 45.8 ms（5.16×） |
| 布局修复测试 | 41 passed |

## 附录 B：commit 对照

| commit | 内容 | 是否需要保留 |
|---|---|---|
| `226df18` | `ParquetDatasetV3` 支持自动统计 + 补齐位监督开关 | ✅ 训练必需 |
| `b7a0115` | TRON2 训练配置 + 修正统计量 + 文档 | ✅ 必需 |
| `d567b98` | CUDA 检查本地放宽 | ⚠️ 本机环境权宜，**不要上游提 PR** |
| `f51b41e` | 加速推理部署配置 | ✅ 部署必需 |
| `e0eb092` | 16→18 动作布局展开 + `state_permutation` | ✅ **部署必需，缺了机器人不会动** |
