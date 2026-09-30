# TRON2 柜内抓取 · PI0.5 LoRA 重建与部署指南 (v2)

> **本文档的目标**：在一台**只有 `tron_ws` 目录的 Ubuntu 20 空白机器**上，
> 从零把训练 + 推理环境搭起来，复现已训好的模型，并接到机器人上。
>
> **v2 说明**：v1（原名 `tron2_cabinet_lora_handover.md`）的所有实测数字都来自
> 原机器（Ubuntu 22.04）。v2 在一台 **Ubuntu 20.04** 机器上**完整复现了一遍**，
> 并修正了 v1 中**会导致部署直接失败的三处配置错误**、补齐了 v1 中**遗漏的
> 两个系统级依赖**。凡标注「🆕 v2 实测」的都是复现时真跑出来的。
>
> **状态**：环境 ✅ 已复现 ｜ 单测 ✅ 41 passed ｜ 推理 ✅ 端到端跑通 ｜ 上机器人 ❌ 未做

---

## 0. 现状速览

| 项目 | 状态 |
|---|---|
| **训练** | ✅ 已完成。17000 步 / 20 小时 27 分 / 最终 loss **0.0088**（比开训降 9.2 倍，全程 0 次尖峰） |
| **代码** | ✅ 已在 git：fork `github.com/Justin186/FluxVLA`，分支 `main` |
| **权重 / 数据** | ⚠️ **不在 git 里**（被 gitignore），需 scp。原机器那份已确认可搬走 |
| **部署代码改动** | ✅ 已落地并测试（41 个单测通过） |
| **环境复现 (v2)** | ✅ Ubuntu 20.04 上完整复现，41 个单测通过 |
| **推理端到端 (v2)** | ✅ 已用**真实数据集观测**跑通，输出 18 维动作，稳态 47.7 ms |
| **真机端到端 (v2)** | ✅ **已用真机实时观测跑通**（只读，未发运动指令）。见 §9.9 |
| **上机器人（真正运动）** | ❌ **未做**。指令报文尚未在真机验证。见 §9.3 / §9.8 |

### 首要提醒

**训练产物（模型 + 数据集）只存在于临时机器上时，先按 §3.3 搬出来，再谈别的。**
→ ✅ v2 已完成搬迁，本机 `/home/lab/tron_ws/` 已持有全部产物，尺寸逐一核对一致（§3.3.4）。

### 🆕 v2 动手前必读：7 处会让部署失败或做错事的问题

按 v1 原样操作，或在 v1 的结论上直接部署，会遇到这些：

| # | 问题 | 症状 | 修复位置 |
|---|---|---|---|
| 1 | pip 装不上 `diffusers` | 8 个镜像全失败，`git clone ... github.com port 443 拒绝连接` | §3.4.1 |
| 2 | flash-attn 官方 wheel 不可用 | `version 'GLIBC_2.32' not found` | §3.4.2 |
| 3 | `import fluxvla` 崩 | 同上的 GLIBC 报错（flash-attn 在**必经导入路径**上） | §3.4.2 |
| 4 | triton 编译失败 | `找不到 -lcuda` | §3.4.3 |
| 5 | ZMQ 服务起不来 / 一推理就崩 | `KeyError: config.themis is required`；`48 vs 135`；`got 32` | §6.2、§6.6 |
| 6 | **冷启动第一条动作是错的** | 右臂最大偏差 0.41 rad（后续 0.03），**确定性复现** | §7.8 |
| 7 | **不切换 `task_description`** | **不报错，但去做错的任务**（5 个任务共用一个模型） | §7.2、§9.6 |

---

## 1. 目标机要求

### 1.1 硬件

| | 要求 | v1 实测环境 | 🆕 v2 复现环境 |
|---|---|---|---|
| GPU | **单张 ≥ 24 GB 显存** | RTX 4090 D，24564 MiB（sm_89） | 2 × RTX 4090，24564 MiB（sm_89） |
| CPU | 建议 ≥ 16 核 | 44 核 | 32 核 |
| 内存 | ≥ 32 GB | 78 GB | 125 GB |
| 磁盘 | **≥ 200 GB 可用** | — | 916 GB 盘，余 267 GB |

**显存为什么必须 24 GB**：micro-batch 4 会 OOM，3 是上限（v1 实测峰值 20538 MiB）。
16 GB 的卡跑不了这个配置，需要下调 batch（见 §5.4）。

**磁盘为什么必须 200 GB**：单个检查点 **29.2 GB**（`.pt` 14.75 GB + `.safetensors` 14.47 GB，
两者内容重复），保留 3 份就是 87.6 GB，加上保存时"先写新的再删旧的"峰值要 **116.8 GB**。
再加上基座权重 6.8 GB。

### 1.2 系统

| | v1 实测环境 | 🆕 v2 复现环境 | 备注 |
|---|---|---|---|
| OS | Ubuntu 22.04.5 LTS | **Ubuntu 20.04.6 LTS** | ⚠️ **差异很大，见 §1.4** |
| 内核 | 5.4.250（velinux） | 5.15.0-139 | |
| NVIDIA 驱动 | **535.154.05** | **575.57.08** | `nvidia-smi` 报 CUDA **12.9** |
| CUDA Toolkit | 未装（用 wheel 自带） | 未装 | 但 `install_env.sh` 会探测 nvcc 决定 profile |

> **[v2 修订] v1 §3.4 提到的 `install_env.sh` 本地补丁在本机不需要。**
> 那段补丁（`git log` 里的 `[Chore] Relax CUDA profile check for local driver`）
> 是为了绕开"驱动报 CUDA 12.2 < profile 要求"的冲突。**只要驱动报的版本 ≥ 12.4，
> 这个检查本来就通过，补丁根本不触发。** 不要为了它专门去改脚本。

### 1.3 软件基线

| | 版本 |
|---|---|
| conda | v1 是 26.7.1 装在 `/opt/miniconda3`；**v2 装在 `/home/lab/miniconda3`，同样能跑** |
| conda env 名 | **`fluxvla`**（必须，见下） |
| Python | 3.10.21 |
| PyTorch | **2.6.0+cu124** |

> **`fluxvla` 这个 env 名字不要改。** 🆕 v2 修订：仓库脚本其实**没有**硬编码
> `/opt/miniconda3/envs/fluxvla/bin/python`（`scripts/zmq_inference_server.sh`
> 只用 PATH 里的 `python`）。真正的要求是 **env 名必须叫 `fluxvla`**，因为
> `scripts/install_env.sh` 内部会 `conda activate fluxvla`。
> conda 装在哪里无所谓。

### 1.4 ⚠️ Ubuntu 20 vs 22 的差异（v1 只提了风险，v2 全部踩到了）

v1 的环境是 22.04（glibc 2.35），目标机是 20.04（glibc 2.31）。**这不是理论风险，
是必然踩到的**：

| 项 | 22.04 (glibc 2.35) | 20.04 (glibc 2.31) |
|---|---|---|
| 官方 flash-attn wheel | ✅ 能 import | ❌ **`GLIBC_2.32 not found`，且它在 `import fluxvla` 的必经路径上** |
| `libcuda.so` 链接符号 | 通常有（装过 cuda-dev） | ❌ **可能只有 `libcuda.so.1`，triton 编译直接失败** |
| GitHub 访问 | — | 视网络而定，**HTTPS 被墙时 git 依赖装不上** |

1. **GLIBC 版本**：20.04 是 glibc 2.31，22.04 是 2.35。
   - 绝大多数 torch wheel 都能跑（实测 `torch==2.6.0+cu124` 官方 wheel 兼容 2.31 ✅）
   - **但 flash-attn 的官方预编译 wheel 是 glibc 2.32+ 的**，必须换构建（§3.4.2）
2. **Python 3.10 需要自己装 conda**，20.04 系统自带是 3.8，不要用系统的。
3. **gcc**：v1 是 gcc 8.3，20.04 默认 gcc 9.4。`requirements` 里的 flash-attn 是预编译
   wheel，正常情况下不需要本地编译。
4. **`/dev/shm`**：如果小于 8 GB，DataLoader 多 worker 会报 `bus error`。
   用 `df -h /dev/shm` 检查（v2 机器是 63 GB，充足）。

---

## 2. 目录布局（关键，不要改）

**所有路径都按这个结构硬编码在配置里。** 换位置就要同时改配置。

```
tron_ws/
├── FluxVLA/                              ← git clone 你的 fork
│   ├── checkpoints/
│   │   ├── pi05_base/                    ← tokenizer 等（⚠️ 见 §3.3.1 说明）
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
│           ├── adapter_model.safetensors ← LoRA 适配器（145 MB）
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
# 装 miniconda（位置随意，v1 用 /opt/miniconda3，v2 用 ~/miniconda3）
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

# 建环境（Python 3.10，名字必须是 fluxvla）
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
```

### 3.2 clone 代码

```bash
mkdir -p ~/tron_ws && cd ~/tron_ws
git clone https://github.com/Justin186/FluxVLA.git
cd FluxVLA
git log --oneline -8
```

> ⚠️ **[v2 实测] 如果 HTTPS 到 GitHub 不通（`Failed to connect to github.com port 443`），
> 但 SSH 通**，直接用 SSH 克隆：
> ```bash
> git clone git@github.com:Justin186/FluxVLA.git
> ```
> 本机就是这种情况：HTTPS 被拒、SSH 认证成功（`Hi Justin186! You've successfully
> authenticated`）。**不要为此换网络，SSH 就够。**

当前 HEAD 应包含这些提交：

```
689e5a5 [Docs] Correct the final-segment loss figure in the handover guide
07ba93f [Docs] Add deployment handover guide for the TRON2 cabinet policy
e0eb092 [Fix] Expand TRON2 policy actions to the robot's 18-dim layout
f51b41e [Feat] Add accelerated deployment config for TRON2 cabinet policy
d567b98 [Chore] Relax CUDA profile check for local driver
b7a0115 [Feat] Add TRON2 cabinet LoRA training config, corrected stats and docs
226df18 [Fix] Support ParquetDatasetV3 in auto statistics and terminal padding
be32110 [Docs] Announce technical report release in multilingual READMEs
```

### 3.3 拉数据与权重（scp / rsync）

> **以下都不在 git 里**（`checkpoints/*`、`datasets/*`、`work_dirs/` 都在 `.gitignore` 内）。

#### 3.3.1 权重（必需）

```bash
# 训练好的模型 —— 合并后的完整模型，推理只需这一个就够
rsync -a --partial root@<源机器>:/root/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/checkpoints/step-017000-epoch-007-loss=0.0088.safetensors \
    ~/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/checkpoints/

# 推理硬断言需要的小文件（5.9 KB）
rsync -a --partial root@<源机器>:/root/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/dataset_statistics.json \
    ~/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/

# LoRA 适配器 + tokenizer（小但重要，续训 / 重合并要用）
rsync -a --partial --include='adapter_model.safetensors' --include='adapter_config.json' \
      --include='llm_backbone_config.json' --include='README.md' \
      --include='tokenizer/' --include='tokenizer/**' --exclude='*' \
      root@<源机器>:/root/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/ \
      ~/tron_ws/FluxVLA/work_dirs/tron2_cabinet_lora_v2/

# 基座权重（6.8 GB）—— ⚠️ 推理也需要，不是只有重训才要（见下方说明）
rsync -a --partial root@<源机器>:/root/tron_ws/FluxVLA/checkpoints/pi05_base_bf16/ \
    ~/tron_ws/FluxVLA/checkpoints/pi05_base_bf16/

# tokenizer 目录（22 MB，不含 model.safetensors）
rsync -a --partial --exclude 'model.safetensors' \
    root@<源机器>:/root/tron_ws/FluxVLA/checkpoints/pi05_base/ \
    ~/tron_ws/FluxVLA/checkpoints/pi05_base/
```

> ⚠️ **[v2 修订] 基座权重推理时也要！** v1 原文写"只有重新训练才需要；纯推理可省"，
> **这是错的**。部署配置里有：
> ```python
> pretrained_name_or_path='./checkpoints/pi05_base_bf16/model.safetensors'
> ```
> 模型是**先按基座构建、再载入合并后的检查点**，缺了它服务起不来。
>
> **[v2 修订] `checkpoints/pi05_base/` 不要整目录拷。** 它里面有个 **14 GB 的 fp32
> `model.safetensors`，没有任何配置引用它**（部署用的是 bf16 那份）。
> 只拷 tokenizer 相关文件（22 MB）即可 —— 上面命令里的 `--exclude` 就是干这个的。
>
> **文件名里有 `=`**，scp 时务必加引号，否则某些 shell 会解析出错。

#### 3.3.2 数据集（2.4 GB）

#### 3.3.3 ⚠️ 数据集里藏着软链陷阱

源机器上每个子集的 `videos/` 目录**不是真目录，是指向绝对路径的软链**：

```
datasets/RealRobot_Tron2_lerobot/lerobot_2026-09-28_22-05-37/videos
  -> /root/tron_ws/datasets_raw/lerobot_2026-09-28_22-05-37/videos
```

后果：
- `du -sh` 只报 **18 MB**，实际是 **2.4 GB**（软链不被计入）
- **直接 `scp -r` 会把软链原样拷过去，到新机器上是断链** → 训练时读不到视频

**正确做法（推荐 rsync 解引用）**：

```bash
# 源机器没有 rsync 的话先装：apt-get install -y rsync
rsync -aL --partial root@<源机器>:/root/tron_ws/FluxVLA/datasets/RealRobot_Tron2_lerobot/ \
      ~/tron_ws/FluxVLA/datasets/RealRobot_Tron2_lerobot/
#      ↑ -L = 解引用，把真实视频文件拷成真目录

# 备选：tar 打包时解引用
tar -czhf tron2_data.tar.gz -C ~/tron_ws/FluxVLA datasets/RealRobot_Tron2_lerobot
```

#### 3.3.4 验证有没有拷对

```bash
cd ~/tron_ws/FluxVLA
du -sh datasets/RealRobot_Tron2_lerobot        # 期望 ~2.4 GB，只报十几 MB 就是软链没解开
find datasets/RealRobot_Tron2_lerobot -type l | wc -l   # 期望 0
find datasets/RealRobot_Tron2_lerobot -name '*.mp4' | wc -l  # 🆕 v2 实测：1668
```

🆕 **v2 实测的完整校验表**（尺寸与源机器逐一比对一致）：

| 产物 | 字节数 |
|---|---|
| `checkpoints/pi05_base_bf16/model.safetensors` | 7,233,650,272 |
| `work_dirs/.../step-017000-epoch-007-loss=0.0088.safetensors` | 14,466,989,776 |
| `work_dirs/.../adapter_model.safetensors` | 145,613,680 |
| 数据集 | 2.4 G / 0 软链 / 1668 mp4 |

**源机器上还有这些，可以不要**：`step-015000` / `step-016000` 两份检查点（已被
step-017000 取代）、step-017000 的 `.pt`（14.75 GB，与 `.safetensors` 内容重复）、
`datasets_raw/`（解引用后不再需要）、`checkpoints/pi05_base/model.safetensors`
（14 GB fp32，无人引用）。

**但建议顺手抢救这两样（不在任何 git 里，机器一释放就没了）**：

```
/root/tron_ws/scripts/          ← 数据制备脚本，见附录 C
/root/tron_ws/TRON2 VLA数据采集资料评估_*.json   ← 14.7 MB
```

### 3.4 装依赖

```bash
cd ~/tron_ws/FluxVLA
conda activate fluxvla

# 关键：real-only 模式 = 训练 + 真实机器人 / 远程推理依赖
bash scripts/install_env.sh real-only --profile cu124
```

**为什么必须 `--profile cu124`**：不指定时脚本按 nvcc 探测，机器上没装 CUDA Toolkit
就可能选错。cu124 的 wheel 在 535+/575+ 驱动上都实测可用。

#### 3.4.1 🆕 [v2 必须] 让 pip 能装 `diffusers`（GitHub HTTPS 被阻断时）

`requirements-base.txt` 第 7 行把 `diffusers` 钉在 git 源上：

```
diffusers @ git+https://github.com/huggingface/diffusers.git@3996788b602eaae4da41a1d45726b62e662b73cf
```

pip 会去 `git clone https://github.com/...`。**HTTPS 不通时，8 个 pip 镜像会全部失败**
（报错是 `fatal: 无法访问 'https://github.com/huggingface/diffusers.git/'：Failed to connect
to github.com port 443`，看起来像镜像坏了，其实跟镜像无关）。

**解法：把 GitHub HTTPS 重写到 SSH。** 用一个 PATH 前置的 `git` 包装脚本，
**不要改全局 git 配置**：

```bash
mkdir -p ~/tron_ws/bin
cat > ~/tron_ws/bin/git <<'EOF'
#!/bin/sh
exec /usr/bin/git -c url."git@github.com:".insteadOf="https://github.com/" "$@"
EOF
chmod +x ~/tron_ws/bin/git

# 验证（应能列出分支而不是报连接失败）
PATH=~/tron_ws/bin:$PATH git ls-remote --heads https://github.com/huggingface/diffusers.git | head -3
```

之后跑安装脚本时带上这个 PATH：

```bash
export PATH=~/tron_ws/bin:$PATH
export GH_PROXY=https://ghfast.top      # flash-attn 的 release 下载也走 GitHub，见 3.4.2
bash scripts/install_env.sh real-only --profile cu124
```

> 为什么要 `GH_PROXY`：`install_env.sh` 下载 flash-attn 时从
> `github.com/.../releases/download/...` 拉，走的是 HTTPS 下载而不是 git。
> 脚本内置了代理候选（默认 `ghfast.top` 等），但显式指定可以跳过失败候选的等待。
> 实测 `ghfast.top` / `gh-proxy.com` / `ghproxy.net` 可用。**只对 Git 依赖无效，
> 那个必须用上面的 SSH 重写。**

#### 3.4.2 🆕 [v2 必须] 替换 flash-attn 为 manylinux 构建

**问题**：官方 release 的 wheel 文件名是
`flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
—— 平台标签是 `linux_x86_64`（在 Ubuntu 22.04 上编的），它的 `.so` 里有：

```
GLIBC_2.32   __libc_single_threaded      ← 唯一一个缺口
```

Ubuntu 20.04 的 glibc 2.31 没有这个符号，`import flash_attn_2_cuda` 直接崩。

**⚠️ 这里有个陷阱：`LD_PRELOAD` 垫片救不了。** 我试过写一个
`char __libc_single_threaded = 0;` 的 shim，**无效** —— 因为这是对 `libc.so.6` 的
**带版本号的符号引用**（verneed 记录在 ELF 里），interpose 一个无版本符号满足不了
版本校验。别在这条路上浪费时间。

**解法：换一个在 manylinux_2_28 容器里构建的 wheel**（glibc 2.28 ≤ 2.31 ✅），
来自社区仓库 `mjun0812/flash-attention-prebuild-wheels`：

```bash
# 直连慢/被墙时前面加 https://ghfast.top
WHEEL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl

curl -L -o /tmp/flash_attn-2.8.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl \
     "https://ghfast.top/${WHEEL}"

# 先验一下 GLIBC 需求（应只出现 2.2.5 / 2.14，不应有 2.32）
unzip -p /tmp/flash_attn-2.8.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl \
      'flash_attn_2_cuda*.so' > /tmp/fa.so
objdump -T /tmp/fa.so | grep -oE 'GLIBC_[0-9.]+' | sort -uV

# 装（文件名必须保持原样，pip 靠文件名解析版本）
pip install --no-deps --force-reinstall \
    /tmp/flash_attn-2.8.3+cu124torch2.6-cp310-cp310-linux_x86_64.whl
```

> ⚠️ 下载后**不要改文件名**。改名成 `fa.whl` 之类会导致
> `ERROR: Invalid wheel filename (wrong number of parts)`。

**验证（必须真跑一次 GPU 计算，不能只看 import）**：

```bash
python -c "
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 256, 8, 64, dtype=torch.bfloat16, device='cuda')
o = flash_attn_func(q, q, q)
print('flash_attn_func:', tuple(o.shape), o.dtype, 'finite=', bool(torch.isfinite(o).all()))
"
```

🆕 v2 实测输出：`flash_attn_func: (2, 256, 8, 64) torch.bfloat16 finite= True`
（版本号显示为 `flash-attn 2.8.3+cu124torch2.6`）

#### 3.4.3 🆕 [v2 必须] 给 triton 补 `libcuda.so`

**问题**：`import fluxvla` 时 triton 会现编一个小 C 库并链 `-lcuda`，但系统上通常
只有 `libcuda.so.1`（dev 软链由 `libcuda1-*-dev` 包提供，没装就没有）：

```
/usr/bin/ld: 找不到 -lcuda
subprocess.CalledProcessError: Command '['/usr/bin/gcc', ... '-lcuda' ...]' returned non-zero exit status 1
```

**方案 A（推荐，需要 sudo）—— 建系统级软链。**

这正是 triton 报错信息本身建议的做法，也是 `libcuda1-dev` 这类 dev 包会做的事：

```bash
sudo ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
```

一条命令解决，**所有 conda 环境、所有需要链接 libcuda 的工具都受益，不需要任何环境变量**。
验证（模拟 triton 实际执行的编译命令）：

```bash
gcc -x c /dev/null -O3 -shared -fPIC -Wno-psabi -o /tmp/probe.so -lcuda -L/lib/x86_64-linux-gnu
echo "exit=$?"     # 期望 exit=0
```

> 🆕 v2 实测：加软链前这条命令报 `找不到 -lcuda`，加完 `exit=0`；
> 清空 `~/.triton/cache` 后在不设任何环境变量的情况下 `import fluxvla` 正常。

**方案 B（无 sudo 时的自包含兜底）** —— triton 支持用 `TRITON_LIBCUDA_PATH`
指定 `libcuda.so` 所在目录，把它指向环境内的一个软链目录，再用 conda 激活钩子
自动导出：

```bash
P=$(conda info --base)/envs/fluxvla
mkdir -p "$P/lib/triton_libcuda" "$P/etc/conda/activate.d"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$P/lib/triton_libcuda/libcuda.so"
echo 'export TRITON_LIBCUDA_PATH="${CONDA_PREFIX}/lib/triton_libcuda"' \
    > "$P/etc/conda/activate.d/fluxvla_triton_libcuda.sh"
```

这样**只要 `conda activate fluxvla` 就自动生效**，不依赖任何手动 export。
代价是它只对这个 conda 环境有效，而且 `conda activate` 之外启动的进程（比如
systemd 服务、IDE 直接调 python）拿不到这个变量。

> **优先用方案 A。** 只有在完全拿不到 root 权限时才退到方案 B。

#### 3.4.4 ⚠️ flash-attn 校验失败会中断脚本，后面两步要手动补

`install_env.sh` 的主流程是：

```
... → install_flash_attn → install_project → verify_project_import → 结束
```

`install_flash_attn` 内部**装完立刻 import 校验**。如果 flash-attn 不可用（比如没做
§3.4.2），脚本带 `set -e` **直接中止**，于是 `install_project`（`pip install -e .`）
和 `verify_project_import` **都不会执行**，`fluxvla` 根本没被装上。

手动补：

```bash
cd ~/tron_ws/FluxVLA
pip install --no-build-isolation -e .
python -c "import fluxvla; print('FluxVLA installed:', fluxvla.__file__)"
```

### 3.5 验证 torch 真能用 GPU

这一步能提前发现 §1.4 的 glibc 问题：

```bash
python -c "
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

期望输出（v1 与 🆕 v2 实测一致）：

```
torch      : 2.6.0+cu124
cuda 运行时: 12.4
GPU        : NVIDIA GeForce RTX 4090
compute cap: (8, 9)
bf16 支持  : True
矩阵乘验证 : True
```

### 3.6 装测试依赖并跑测试

```bash
pip install pytest
cd ~/tron_ws/FluxVLA
python -m pytest test/test_transforms/ test/test_datasets/ -q
# 期望：41 passed   （v1 ✅ / 🆕 v2 ✅ 均为 41 passed）
```

---

## 4. 环境版本清单（实测）

```bash
python -c "
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

| 包 | v1 实测 | 🆕 v2 实测 | 说明 |
|---|---|---|---|
| torch | 2.6.0+cu124 | 2.6.0+cu124 | |
| torchvision | 0.21.0+cu124 | 0.21.0+cu124 | |
| transformers | 5.3.0 | 5.3.0 | |
| numpy | **1.26.4** | 1.26.4 | ⚠️ 必须 1.x，2.x 会和 mmengine/opencv 冲突 |
| mmengine | 0.10.7 | 0.10.7 | 配置系统 |
| accelerate | 0.33.0 | 0.33.0 | |
| diffusers | 0.37.0.dev0 | 0.37.0.dev0 | 开发版，正常 |
| timm | 0.9.10 | 0.9.10 | |
| einops | 0.4.1 | 0.4.1 | |
| **peft** | **0.19.1** | 0.19.1 | LoRA 必需 |
| safetensors | 0.8.0 | 0.8.0 | |
| pyarrow | 24.0.0 | 25.0.1 | 差异无影响 |
| pandas | 2.3.3 | 2.3.3 | |
| **av** | **14.2.0** | 14.2.0 | 视频解码 |
| **torchcodec** | **0.2.1** | 0.2.1 | ⚠️ **必须和 torch 配对**：torch 2.6 → 0.2.1，torch 2.8 → 0.7.0 |
| imageio | 2.37.3 | 2.38.0 | 差异无影响 |
| cv2 | 4.11.0 | 4.11.0 | |
| **triton** | **3.2.0** | 3.2.0 | 加速推理必需 |
| **flash_attn** | **2.8.3** | 2.8.3 | ⚠️ v2 用的是**社区 manylinux 构建**（§3.4.2） |
| wandb | 0.21.0 | 0.21.0 | 装了但**不要开**，见 §7.5 |
| zmq | 27.1.0 | 27.2.0 | 差异无影响 |
| msgpack | 1.2.1 | 1.2.3 | 差异无影响 |
| PIL | 12.3.0 | 12.3.0 | |

**视频解码后端**（`fluxvla/datasets/utils/video_decode.py`）：优先用 **torchcodec**，
失败回退到 torchvision。版本对不上会直接抛
`matching PyTorch (torch 2.8: torchcodec 0.7.0; torch 2.6: torchcodec 0.2.1)`。

---

## 5. 训练

### 5.1 启动命令

```bash
cd ~/tron_ws/FluxVLA
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

torchrun \
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
| `ori_action_dim` | **16** | 真实（未补齐）动作宽度，推理截断靠它 |
| `enable_gradient_checkpointing` | **True** | 实测 ON/OFF 显存与速度完全相同，开着当保险 |
| `active_trackers` | **`('jsonl',)`** | 见 §7.5 |

**实测性能**：稳态 **4.21 s/步** → 17000 步 ≈ **20 小时**，峰值显存 **20538 MiB**。

### 5.3 训完的验证标准

```bash
python -c "
import json, statistics, glob
f = sorted(glob.glob('work_dirs/tron2_cabinet_lora_v2/pi05_paligemma_tron2_cabinet_lora_*.jsonl'))
ls = [json.loads(l) for l in open(f[-1]) if l.strip()]
loss = [x['VLA Train/Loss'] for x in ls]
print('步数         :', len(ls), '(期望 17000)')
print('全程中位数   : %.4f (期望 ~0.011，含早期高 loss)' % statistics.median(loss))
print('末段1000步   : %.4f (期望 ~0.008)' % statistics.median(loss[-1000:]))
print('最后 lr      :', ls[-1]['VLA Train/Learning Rate'], '(期望 2.5e-06)')
print('尖峰(loss>1) :', sum(1 for l in loss if l > 1), '次 (期望 0)')
"
```

**v1 结果**：17000 步 / 全程中位 0.0113 / 末段 1000 步 **0.0081** / lr 2.5e-06 / 尖峰 0 次。
🆕 **v2 用搬过来的 jsonl 复算，数字完全一致** —— 这同时证明了搬迁的数据是完整的。

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

`configs/pi05/pi05_paligemma_tron2_cabinet_lora_deploy.py`

它用 mmengine 的 `_base_` 复用训练配置，**覆盖 `inference_model`** 换成
Triton + CUDA Graph 加速版，并定义 ZMQ 服务所需的 `themis` 段。

### 6.2 启动服务

```bash
cd ~/tron_ws/FluxVLA
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

scripts/zmq_inference_server.sh \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora_deploy.py \
  --ckpt-path work_dirs/tron2_cabinet_lora_v2/checkpoints/step-017000-epoch-007-loss=0.0088.safetensors \
  --port 5555
```

> 🆕 **[v2 修订] v1 的这条命令跑不起来。** `FluxVLAZMQEvalServer` 在
> `build_zmq_eval_server_from_config` 里**强制要求 `themis` 段**，而 v1 的部署配置里
> 没有，直接 `KeyError: config.themis is required`。v2 已把 `themis` 段补进部署配置，
> 所以现在这条命令可以**原样使用**。

**启动后必须先 warmup 一次再让机器人动。** 第一次调用要做 Triton JIT 编译 +
CUDA Graph 捕获，实测：

| | v1 | 🆕 v2 |
|---|---|---|
| 首次调用 | 29.3 s | **5.6 s** |
| 稳态 | 45.8 ms | **47.7 ms**（20 次中位） |

首次调用日志里会打：

```
[Triton Inference] Recording CUDA Graph ...
[Triton Inference] CUDA Graph recorded successfully!
```

> ⚠️ 🆕 **v2 附加警告：warmup 不只是为了延迟，更是为了正确性。**
> 修复前的版本里，**冷启动第一次推理的输出是错的**（详见 §7.8）。
> 修复已包含在本仓库中，但**如果你用的是修复前的提交，务必 warmup 并丢弃第一次结果**。

### 6.3 实测延迟

测法：加载 `step-017000-*.safetensors`，`torch.autocast('cuda', dtype=torch.bfloat16)`，
warmup 后计时 20 次取中位数。

| 路径 | 稳态延迟 | 占预算 | 加速比 |
|---|---|---|---|
| 朴素 `PI05FlowMatching` | 236.3 ms（230.6 ~ 237.7） | 22.1% | 1× |
| **加速 `PI05FlowMatchingRTCInference`** | **45.8 ms**（45.7 ~ 45.9） | **4.3%** | **5.16×** |

🆕 **v2 端到端实测**（走完整 ZMQ + 反归一化链路，prompt 142 token）：

```
20 次中位: 56.2 ms wall / 48.4 ms 服务端
```

与 v1 的 45.8 ms 基本一致，**说明 §6.4 的加速比结论成立**。

**延迟预算不是 33 ms，是 1067 ms** —— 一次推理输出 32 步动作，
机器人按 30 Hz 执行：`32 ÷ 30 = 1.067 秒`。

**结论：延迟完全不是瓶颈**，连朴素路径都只占 22%。

> ⚠️ **加速比是 5.16×，不是 `docs/inference_acceleration.md` 里写的 15×。**
> 那个 15× 是在 **A100** 上标的，sm_89（4090）实测只有 5.16×。

### 6.4 ⚠️ 必须用 RTC 版加速类

```
PI05FlowMatchingInference        ✗ RuntimeError: tensor a (2) vs b (3)   ← 3 路相机报错
PI05FlowMatchingRTCInference     ✓ 正常
```

**`PI05FlowMatchingInference`（非 RTC）在 3 路相机下会崩。**
`docs/inference_acceleration.md` 里的 π0.5 范例写的是非 RTC 版，
**直接照抄会在我们的 3 相机配置上失败**。部署配置里已经用的是 RTC 版。

### 6.5 🆕 [v2 新增] `predict_action` 的观测契约（实测，v1 完全没写）

这是**最容易踩的一块**，v1 只写了模型张量形状，没写 ZMQ 请求里该放什么。

ZMQ 请求用 msgpack，端点 `predict_action`，`data` 里是关键字参数直接喂给
`FluxVLAZMQEvalServer._handle_predict`：

```python
{
  "endpoint": "predict_action",
  "data": {
    "observation": {
        # ★ 16 维，【策略顺序】[L7, gripL, R7, gripR]
        "qpos":   np.float32[16],

        # ★ 18 维，【机器人原生顺序】[L7, R7, head(2), gripL, gripR]
        "states": np.float32[18],

        "cam_high":        np.uint8[H, W, 3],
        "cam_left_wrist":  np.uint8[H, W, 3],
        "cam_right_wrist": np.uint8[H, W, 3],

        # ★ 必须是 5 个训练任务之一，逐字一致。这是唯一的任务路由手段
        "task_description": "Press the red button",
    },
    "unnorm_key": "private",
    "episode_id": "...",   # 换了 episode 会重置历史
    "seed": 7,
    "reset": True,
  }
}
```

**两个字段的来源与量纲**（v2 在真机上验证过，见 §9.4）：

| 字段 | 来自 | 换算 |
|---|---|---|
| `qpos[0:7]` | `/joint_states` 前 7 维（左臂） | 直接用 |
| `qpos[7]` | `request_get_limx_2fclaw_state` 的 `left_opening` | **÷100**（hw 0-100 → 0-1） |
| `qpos[8:15]` | `/joint_states` 第 8-14 维（右臂） | 直接用 |
| `qpos[15]` | `right_opening` | **÷100** |
| `states[0:14]` | `/joint_states` | 直接用 |
| `states[14:16]` | `/joint_states` 最后两维（头部） | 直接用 |
| `states[16:18]` | 夹爪 | **÷100** |

响应（`MsgSerializer` 解码后）：

```python
{"ok": True, "actions": np.float32[50, 18], "action_horizon": 50,
 "action_dim": 18, "denormalized": True, "inference_time_s": 0.0477, ...}
```

**为什么必须分成 `qpos` 和 `states` 两个字段**（这是 v1 没讲清、也是踩坑最多的地方）：

| 字段 | 用途 | 为什么是这个维度 |
|---|---|---|
| `qpos` | 喂给 `PrivateInferenceDataset` → `NormalizeStatesAndActions` | 归一化统计量 `dataset_statistics.json['private']['proprio']` **只有 16 维**（数据集 `observation.state` 就是 16 维，不含头部）。归一化发生在补齐到 32 维**之前**，所以送 18 维会直接 `operands could not be broadcast together with shapes (18,) (16,)` |
| `states` | 供 `DenormalizeDeltaAction` 做 delta 还原 + `DenormalizeTron2Action.current_head` 读头部 | `state_permutation` 长度校验必须是 18；`current_head` 直接读 `state[14:16]` |

`PrivateInferenceDataset` 不设置 `last_raw_state`，所以 `ros_server` 会回落到
`observation['states']`（`ros_server.py:208`）—— 这就是 `states` 这个字段存在的理由。

**如果只传 `qpos`（18 维），会依次遇到**：

```
ValueError: operands could not be broadcast together with shapes (18,) (16,)   # 归一化
```
（就算绕过归一化，接着还会）
```
ValueError: Current raw robot state is required to restore delta actions.
```

### 6.6 🆕 [v2 修正] 部署配置里两个必须改的值

v1 的部署配置有两处**会直接导致推理崩溃**的值，v2 已修好：

| 参数 | v1 值 | 正确值 | 不改的后果 |
|---|---|---|---|
| `triton_max_prompt_len` | `48` | **`200`** | `RuntimeError: The size of tensor a (48) must match the size of tensor b (135)` |
| `ori_action_dim` | `14` | **`16`** | `ValueError: Expected a 16-dim policy action or an already expanded 18-dim action, got 32` |

**为什么 `triton_max_prompt_len=48` 太小**：`PreparePromptWithState` 会把
**32 个**（补齐后的）归一化状态值离散成 256 桶写进 prompt 文本：

```
Task: Press the red button, State: <32 个数>;\nAction: 
```

实测 token 数（`checkpoints/pi05_base` 的 tokenizer）：

| 状态维度 | token 数 |
|---|---|
| 16 维 | ~78 |
| **32 维（实际用的）** | **~135–142** |

`ProcessPrompts.max_len=200` 是硬上限，所以 `triton_max_prompt_len` 设 **200** 才安全。

> 顺带澄清一个容易误读的地方：`PreparePromptWithState` 的 docstring 说"padding 应在
> tokenize 之后，对齐 OpenPI"——**这句是过时的**。训练配置和推理配置的 transform
> 顺序**都是**先 `NormalizeStatesAndActions(state_dim=32)` 再 `PreparePromptWithState`
> 再 `ProcessPrompts`，两边一致，所以模型见到的 prompt 形态没有偏差。

**为什么 `ori_action_dim` 必须改**：模型输出 `max_action_dim=32` 维（补齐），
而 `DenormalizeTron2Action` 只接受 16 或 18 维。平时这一步截断是在 **action head** 里
做的（`flow_matching_head.py` / `flow_matching_inference_head.py` 里的
`actions[..., :self.ori_action_dim]`），**但 Triton/RTC 加速路径绕过了 head**，
所以 v2 在 `PI05FlowMatchingRTCInference.predict_action` 里补了同样的截断。
`14` 是 π0.5 基座的旧值，TRON2 是 16。

---

## 7. 必须知道的坑

### 7.1 动作布局：模型 16 维 vs 机器人 18 维

| | 布局 | 夹爪位置 | 维度 |
|---|---|---|---|
| **模型** | `[左臂7, 左夹爪, 右臂7, 右夹爪]` | 7, 15 | **16** |
| **机器人** | `[左臂7, 右臂7, 头部2, 左夹爪, 右夹爪]` | 16, 17 | **18** |

**已于本仓库修复**：`fluxvla/transforms/normalize.py::DenormalizeTron2Action`，
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
而机器人原始状态是 18 维。写 16 直接抛错。

🆕 **v2 实测验过布局没错位**：把 `states[14:16]` 设成 `[0.7, -0.3]` 送进去，
输出的 idx 14-15 精确等于 `[0.7, -0.3]`，且右臂 idx 7-13 里**没有**混进 0.7 ——
正是 §7.1 说的"错位一格并混入头部值"的反向验证。

**验证**：`test/test_transforms/test_tron2_action_layout.py`（11 个用例）。
其中两条是防回退的：长度 16 的 permutation 必须报错、
16 维动作必须让 runner 的 `actions[:, 16]` 抛 `IndexError`。

### 7.2 左臂 + 左夹爪退化，但**右夹爪不是**

**要按"左臂和左夹爪不可用、右夹爪可用"来设计部署。**

| 通道 | 状态 | 证据 |
|---|---|---|
| 左臂 7 维 | **退化** | 全部 56888 帧左臂关节恒定 |
| **左夹爪** | **退化，恒为 0** | 统计量 `q01=0 q99=0 min=0 max=0 std=0` |
| **右夹爪** | **✅ 有效，必须用** | 见下方 v2 实测：两个拧旋钮子集里跑到 0.94 |

后果：
- 模型对左臂/左夹爪**没有有效学习信号**
- 左夹爪量化区间退化 → **反归一化把任何输入都塌成 0**，输出恒为 0

🆕 **v2 端到端复现了这一点**：50 步动作里 `actions[:, 16]`（左夹爪）min=max=**0.0**；
左臂 7 维在这个 chunk 内的变化幅度只有 ~0.004（因为它本质上是"当前位姿 + 极小 delta"，
delta 还原后自动贴着当前位姿）。**这也说明 §7.1 的 delta 还原是对的**——
送进去的真实状态是 `[-0.0203, 0.2433, -0.0065, -1.4265, 0.1752, -0.193, -0.0214, ...]`，
第一步左臂输出 `[-0.0266, 0.2413, -0.0038, -1.4286, 0.1747, -0.1925, -0.0162]`，
右臂 max|Δ| = 0.032。

#### ⚠️ [v2 重要修正] 右夹爪不是退化的 —— 2 个任务需要它

> 这条最初被漏掉了：**训练不是只做了"按按钮"，是 5 个任务**，其中 2 个拧旋钮的任务**要用夹爪**。

按子集实测右夹爪的范围（`observation.state` 第 15 维）：

| 子集 | 帧数 | 任务 | gripR min/max |
|---|---|---|---|
| `22-05-37` | 8834 | Press the red button | 0.0000 / **0.0000** |
| `22-26-43` | 8242 | Press the black button | 0.0000 / **0.0000** |
| `22-36-10` | 8097 | Press the green button | 0.0000 / **0.0000** |
| `22-44-32` | 17875 | **Switch the selector from left to right** | 0.0100 / **0.9400** |
| `22-54-05` | 13437 | **Switch the selector from right to left** | 0.0100 / **0.9400** |

而且**同一个模型、同一套观测，只换 `task_description`，输出行为就变了**（v2 实测）：

| task_description | 右夹爪 idx17 min/max |
|---|---|
| `Press the red button` | 0.0039 / **0.0469**（不动） |
| `Switch the selector from left to right` | 0.0254 / **0.8223**（张开去抓） |

**两条硬结论**：
1. **夹爪必须下发**，否则拧旋钮任务做不了。量纲已确认（§9.5）。
2. **`task_description` 必须按当前任务切换** —— 这是 5 个任务共用同一个模型的**唯一路由手段**。
   发错文本会让策略去做错事（比如拿着"按按钮"的指令去执行拧旋钮场景）。
   ⚠️ 这个坑比夹爪本身更危险，因为**不报错、只是做错**。

**部署建议**：锁定左臂（客户端把左臂 7 维替换成实测位姿）；左夹爪不下发；
右夹爪按任务下发；任务文本由操作员显式选择（见 §9.6）。

### 7.3 stats 必须用修正版

**必须用 `datasets/RealRobot_Tron2_lerobot/tron2_stats_armsymmetric.json`**
（已提交到 git），不能用自动统计的版本。

原因（`dataset_statistics.json` 里的 `_fix_note` 原文）：自动统计下左臂维度会按
**约 150 倍**的尺度反归一化 → **左臂指令直接飙飞**。这个文件是**手工修正**过的产物，
无法自动重生成 —— 这也是它被强制加进 git（原本在 `datasets/*` 的 gitignore 里）的原因。

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

当前是 `17000 % 1000 == 0` ✅。

**磁盘峰值约束**（保存是"先写新的、再删旧的"，峰值多占一份）：

| `max_keep_ckpts` | 稳态 | 峰值 | 200 GB 盘上的余量 |
|---|---|---|---|
| 2 | 58.4 GB | 87.6 GB | 充足 |
| **3（当前）** | **87.6 GB** | **116.8 GB** | **~83 GB** |
| 4 | 116.8 GB | 146.0 GB | 很紧张 |

### 7.7 环境残留可能有 PYTHONPATH

如果目标机之前装过东西，**跑训练前 `unset PYTHONPATH`**。

我们踩过这个坑：残留的 `PYTHONPATH` 让 Python 导入了系统的 `send2trash`，
导致**每次删检查点都进回收站而不是真删**，磁盘悄悄涨了 28 GB。清理：

```bash
/bin/rm -rf ~/.local/share/Trash/files/*
df -h /
```

### 7.8 首次推理要先 warmup（🆕 修好前它会返回错的动作）

见 §6.2。**服务起来后必须先 warmup 一次。**

🆕 **v2 实测发现了一个会让机器人首条指令出错的 bug**（已在本仓库修复）。

**现象**：冷启动后连发 5 次**完全相同**的请求，第 1 次的输出明显跑偏：

| 调用次序 | 右臂 max\|Δ\|（相对当前位姿） |
|---|---|
| **第 1 次（刚启动）** | **0.4132** ❌ |
| 第 2 次 | 0.0285 |
| 第 3 次 | 0.0333 |
| 第 4 次 | 0.0323 |
| 第 5 次 | 0.0357 |

**这是确定性的，不是随机噪声**：两次独立冷启动得到的第 1 次结果几乎一模一样
（`0.4132` / `0.4132`，右臂向量 `[-0.3572, -0.06, 0.404, -1.3222, ...]`）。
0.41 rad 的偏差足以让机械臂做出错误动作。

**根因**（`fluxvla/models/vlas/pi05_flowmatching_inference_rtc.py`）：
`_triton_forward()` 在 CUDA Graph 未就绪时调用 `_build_cuda_graph()`，而后者会
**先跑 3 次 eager 预热 + 1 次 capture**，全部是**就地**改写
`diffusion_noise` 这个去噪缓冲区。返回后紧接着 `replay()` 一次 —— 于是第一次调用
返回的是**多次串联去噪**的结果，而不是单次去噪。

**修复**：`_build_cuda_graph()` 之后重新走一遍 `_triton_forward()`，把调用方原始输入
重新写进缓冲区再 replay。修复后第 1 次调用的 max|Δ| 回到 **0.0394**，与后续调用一致。

**修复不代表可以省掉 warmup** —— 首次调用那 5.6 s（Triton JIT + Graph 捕获）依然存在，
只是现在它的**输出也是对的**了。

> 🆕 顺带一个现象：后续每次调用的结果彼此略有差异（上表 0.0277 ~ 0.0357 波动），
> 这是正常的采样噪声 —— `predict_action` 里 `noise=None` 会走 `torch.randn`，
> 而服务端 `themis.ros_server.forward_seed` 默认是 `False`，所以请求里的 `seed`
> **不生效**。要复现同一结果需要把 `forward_seed` 打开。

### 7.9 🆕 腕部相机首帧是黑的

采集时腕部相机有启动延迟，**首帧可能是黑的**。

v1 说"1~6 帧（33~200 ms）"，🆕 **v2 实测比这个更严重**：数据集里
`cam_left_wrist` / `cam_right_wrist` 的**第 0 帧 `mean=0.0`（纯黑）**，跳过 10 帧后才正常
（`cam_high` 首帧正常）。如果服务端拿黑帧推理，输出不可信。

**建议**：客户端加"首帧有效再推理"的检查（比如判 `frame.mean() > 1`，或等
首帧到齐后再发请求）。否则机器人第一次推理会基于黑图。

### 7.10 🆕 仓库里有一个"旧版 ZMQ 服务"容易混淆

`fluxvla/engines/runners/serving/` 下有两个 ZMQ 实现，**协议不一样**：

| 文件 | 服务类 | 观测字段 | 谁在用 |
|---|---|---|---|
| `zmq_eval_server.py` | `FluxVLAZMQEvalServer` | `observation`（含 `qpos`+`states`） | ✅ **`scripts/zmq_inference_server.py` 用的就是它** |
| `zmq_server.py` | 另一套 | `obs_data`，且把 `qpos` 当 18 维原始状态 | 未在文档路径中使用 |

同样，`fluxvla/engines/runners/serving/serializers.py` 里的
`encode_predict_request()` 是给**旧的那套**用的；发给
`FluxVLAZMQEvalServer` 会得到
`got an unexpected keyword argument 'obs_data'`。

**给当前服务写客户端时，直接构造上面 §6.5 的 dict，用 `MsgSerializer` 编解码。**

---

## 8. 故障排查速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `'PureOverwatch' object has no attribute 'local_rank'` | 用 `python` 直跑 | 改用 `torchrun`（§5.1） |
| `Failed to connect to github.com port 443`（pip 装 diffusers） | GitHub HTTPS 被阻断 | git SSH 重写（§3.4.1） |
| `GLIBC_2.32 not found`（flash_attn_2_cuda） | 官方 wheel 是 glibc 2.32+ 构建 | 换 manylinux 构建（§3.4.2）。**LD_PRELOAD 无效** |
| `/usr/bin/ld: 找不到 -lcuda` | 缺 `libcuda.so` dev 软链 | `sudo ln -s libcuda.so.1 libcuda.so`；无 sudo 时用 `TRITON_LIBCUDA_PATH`（§3.4.3） |
| `install_env.sh` 在 flash-attn 处中止，之后没有项目安装 | `set -e` 中断了后续步骤 | 手动 `pip install -e .`（§3.4.4） |
| `CUDA out of memory`（训练） | batch 太大 | 降到 2 或 1（§5.4） |
| `CUDA out of memory`（保存时） | **有别的进程占着显存** | `pkill -f scripts/train.py`，确认 `nvidia-smi` 干净再启 |
| `dataset_statistics file not found at ...` | 层级不对 | 放到 `ckpt_path/../..`（§2） |
| `KeyError: config.themis is required` | 部署配置缺 `themis` 段 | 用 v2 的部署配置（§6.2） |
| `The size of tensor a (48) must match ... (135)` | `triton_max_prompt_len` 太小 | 改成 200（§6.6） |
| `Expected a 16-dim policy action ... got 32` | `ori_action_dim` 没生效 / 值错 | 改成 16 且确认 RTC 类做了截断（§6.6） |
| `operands could not be broadcast ... (18,) (16,)` | 把 18 维状态当 `qpos` 传了 | `qpos` 传 16 维，另用 `states` 传 18 维（§6.5） |
| `... unexpected keyword argument 'obs_data'` | 用了旧版服务的请求格式 | 见 §7.10 |
| `state_permutation must contain every index` | permutation 写成 16 长度 | 改成 18（§7.1） |
| `IndexError` 在 `actions[:, 16]` | transform 没换成 `DenormalizeTron2Action` | 见 §7.1 |
| `tensor a (2) vs b (3)` | 用了非 RTC 加速类 | 换 `PI05FlowMatchingRTCInference`（§6.4） |
| `matching PyTorch (torch 2.8: torchcodec 0.7.0 ...)` | torchcodec 版本不配对 | torch 2.6 → torchcodec 0.2.1 |
| 训练卡住不动、日志无输出 | wandb 在等登录 | `export WANDB_MODE=disabled`（§7.5） |
| 磁盘莫名增长 | 删的东西进了回收站 | `unset PYTHONPATH` + 清回收站（§7.7） |
| 视频读不到 / 帧全黑 | 数据集软链没解开 | §3.3.3 |
| 首帧推理结果异常 | 腕部相机首帧是黑的 | 跳过前 10 帧（§7.9） |
| 冷启动后**第一条**动作明显偏大（~0.4 rad） | CUDA Graph 捕获前没复位去噪缓冲区 | 用本仓库修复后的版本；旧版本必须丢弃第一次结果（§7.8） |
| 同样的输入每次输出都不同 | `forward_seed` 默认关闭，`seed` 不生效 | 正常采样噪声；要复现需开 `forward_seed`（§7.8） |
| **机器人做错的任务**（不报错） | `task_description` 发错 | 必须从 5 个训练任务里逐字选（§9.6） |
| 红蓝颜色看起来反了 / 策略行为诡异 | 相机通道顺序错（BGR 当 RGB 用） | `cv2.imdecode` 后 `[..., ::-1]`（§9.4） |
| 夹爪读数像 0.02 或 2 | 忘了 `/100`（hw 0-100 ↔ 数据集 0-1） | 见 §9.5 |
| 拧旋钮任务夹爪不动 | 夹爪指令没下发 | 需要 `--send-gripper`（§7.2） |

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

## 9. 接机器人（🆕 v2 新增）

### 9.1 网络

机器人自带一个**内部局域网**，插网线后由机器人侧 DHCP 分配地址。**不需要手动设静态 IP。**

| 主机 | 地址 | 说明 |
|---|---|---|
| 网关 / 路由器 | `10.192.1.1` | 也是 IPv6 RA 的发送方 |
| 运控电脑（主控） | `10.192.1.2` | 管理页 `http://10.192.1.2:8080`；遥操作 `:5000` / `:9191`；ROS master `:11311` |
| 开发拓展模块 | `10.192.1.4` | `ssh guest@10.192.1.4`，密码 `123456` |
| PC（本机） | DHCP 分配，🆕 实测拿到 `10.192.1.154/24` | 文档里建议静态 `10.192.1.120`，但 DHCP 实测可用 |

**排障要点（🆕 实测踩过）**：

- **症状**：网线插着、`enp5s0` 是 UP 且 1000 Mbps 全双工，但 `ping 10.192.1.4` 不通。
- **原因**：`enp5s0` 被绑定到一个叫 `lab_xiao` 的 NetworkManager 配置上，
  **静态写死 `172.21.15.31/24`**（另一个实验室的网段），路由表里根本没有
  `10.192.1.0/24`，于是 ping 走了 WiFi 的默认路由出去。
- **判断方法**：
  ```bash
  ip -br addr                      # 看有线口是不是错的网段
  ip route                         # 有没有 10.192.1.0/24
  nmcli device status              # 有线口绑的是哪个 profile
  nmcli con show <profile>         # 看 ipv4.method / ipv4.addresses
  ```
- **修复**（不影响原有配置，另建一个）：
  ```bash
  nmcli con add type ethernet ifname enp5s0 con-name tron2_robot \
        ipv4.method auto ipv6.method auto
  nmcli con up tron2_robot
  ip -br addr show enp5s0          # 应拿到 10.192.1.x/24
  ping -c 3 10.192.1.2
  ping -c 3 10.192.1.4
  ```
- ⚠️ **别乱改成静态**：如果照文档设 `10.192.1.120/24` 但把网关留空，
  会丢掉机器人网的默认路由。DHCP 已经能工作，优先用它。
- **两条有线配置可以共存**：原来的 `lab_xiao`（静态 `172.21.15.31/24`）仍然保留，
  autoconnect 也是开的。NetworkManager 在两个配置优先级相同时会选**最近一次激活**
  的那个，所以表现是"换到哪里就自动跟着切"。若要强制优先用机器人网：
  ```bash
  nmcli con mod tron2_robot connection.autoconnect-priority 10
  ```
  （副作用：插回实验室网时会先等一次 DHCP 超时再回落到 `lab_xiao`。）
- **WiFi 不会断**：有线口拿到的是 metric 20100 的默认路由，WiFi 是 600，
  外网仍走 WiFi（🆕 实测改完后 tuna 200、ping 8.8.8.8 79 ms，外网正常）。

### 9.2 上机器人之前的清单

| # | 项 | 状态 |
|---|---|---|
| 1 | 真实观测跑一次推理（数据集真实帧 + 真实关节） | ✅ **已做**（§6.5 / §7.2） |
| 2 | 左臂安全钳制 | ✅ **已做**（§9.7，客户端把左臂锁成实测位姿） |
| 3 | 腕部相机首帧检查 | ⚠️ 已确认真机首帧**不是**黑的（§9.4），但客户端还没加"首帧有效"断言 |
| 4 | `prepare_pose` 与采集初始位姿一致性 | ❌ 未做，需现场确认 |
| 5 | 机器人能访问本机推理端口 | ✅ **已做**并验证（§9.1 防火墙） |
| 6 | 用真机实时数据跑通端到端 | ✅ **已做**（§9.9） |

### 9.3 还没验证过的

- **真机上真正运动过** —— 到 v2 为止**一次运动指令都没发过**，全部是只读。
  指令报文格式来自仓库实现（§9.5），**尚未在真机验证**。
- **抓取/拧旋钮的成功率** —— 没有评测环境。
- **Orin 板端部署** —— `docs/orin_docker_runtime.md` 有官方文档，未在 Orin 上验证。
- **ROS 直连路径（`Tron2Operator` + `tron2_inference_runner.py`）** ——
  v2 验证的是 ZMQ 路径。注意 `Tron2Operator` 对相机用的是
  `/camera/*/color/image_rect_raw`（raw），而数据采集用的是
  `image_resized`/`image_raw` 的 **compressed** 版本（§9.4），**两者不是同一个流**，
  接 ROS 路径时要注意分辨率与压缩差异。

---

### 9.4 🆕 数据源（v2 真机实测）

机器人自带一个 `10.192.1.0/24` 内网，插网线后由机器人侧 DHCP 分配地址（§9.1）。

| 用途 | 端点 | 说明 |
|---|---|---|
| **图像** | `http://10.192.1.4:8770/frame/{top,left,right}` | `.4` 上跑 `~/camera_shim.py`（纯订阅，不控制）。源码在机器人上，非仓库代码 |
| **关节 16 维** | `ws://10.192.1.2:5000` → `request_get_joint_state` | 返回 `q` = `[L7, R7, head_pitch, head_yaw]`，与 ROS `/joint_states` **逐位相同** |
| **夹爪** | 同 WebSocket → `request_get_limx_2fclaw_state` | 返回 `left_opening` / `right_opening`，**0-100** |
| 机器人信息 | `notify_robot_info`（3 Hz 主动推送） | 电量/固件/`imu`/`motor`/`tele_operation` |
| 管理页 | `http://10.192.1.2:8080` | LimX Studio |

**相机键映射**（`.4` 的前端代码 `imageTopicUtils.ts` 里 `HEAD_CAMERA_KEYWORDS = ['top','high','head']`）：

```
top   -> /camera/top/color/image_raw/compressed              -> cam_high
left  -> /camera/left/color/image_resized/compressed         -> cam_left_wrist
right -> /camera/right/color/image_resized/compressed        -> cam_right_wrist
```

实测三路都是 **480×640、~30 Hz**，与数据集视频分辨率一致。

> ⚠️ 🆕 **通道顺序是 RGB 还是 BGR？** ROS 话题的 `format` 字段写着
> `rgb8; jpeg compressed bgr8` —— **发布端压成 bgr8 再 JPEG**。
> 所以 `cv2.imdecode()` 拿到的是 **BGR**，而训练时视频按 **RGB** 解码。
> **客户端必须 `[..., ::-1]` 换成 RGB**，否则红蓝互换、策略会静默劣化（不报错）。

🆕 **腕部相机首帧在真机上不是黑的**（`cam_left_wrist` mean≈128、`cam_right_wrist` ≈133），
和数据集中那 1-10 帧黑帧的现象不同 —— 说明那是采集软件启动时的问题，
不是相机本身。仍然建议客户端加"首帧有效"断言。

### 9.5 🆕 指令接口（从仓库实现中提取，官方文档有出入）

| 用途 | 请求 | 报文 |
|---|---|---|
| 手臂 14 关节 | `request_movej` | `{'joint': [L7,R7], 'time': 秒}` |
| **手臂+头 16 关节（高频）** | `request_servoj` | 见下 |
| 头部 | `request_moveh` | `{'joint': [pitch, yaw], 'time': 秒}` |
| 夹爪下发 | `request_set_limx_2fclaw_cmd` | `{'right_opening': 0-100, 'right_speed': 100, 'right_force': 100}` |
| 夹爪读取 | `request_get_limx_2fclaw_state` | 返回 `*_opening` 0-100 |
| 急停 | `request_emgy_stop` | **仅空闲模式有效（运动中无效）** |

**`request_servoj` 的 16 维 = 14 臂 + 2 头**（`tron2_operator.py:392` 的 docstring 明写
`ServoJ requires 16-dim commands: 14 arm joints + 2 head joints`），即
`[L7, R7, head_pitch, head_yaw]` —— 和 `/joint_states` 同序。

> ⚠️ **官方 SDK 文档里那个 `{filter_ratio, q}` 报文和仓库实现不一致。**
> 仓库用的是（`tron2_operator.py:457`）：
> ```python
> {'q': q16, 'v': [0.0]*16, 'kp': kp16, 'kd': kd16,
>  'tau': [0.0]*16, 'mode': [0]*16, 'na': 0}
> ```
> 两套哪个对**没在真机验证过**（§9.8 的自检就是干这个的）。

**正增益**（`tron2_operator.py:148`，顺序为 abad/hip/yaw/knee/wrist_yaw/wrist_pitch/wrist_roll）：

```python
kp = [420, 420, 300, 300, 200, 200, 200,   # 左臂
      420, 420, 300, 300, 200, 200, 200,   # 右臂
       60,  60]                            # 头部
kd = [ 12,  12,  15,  15,  10,  10,  10,
       12,  12,  15,  15,  10,  10,  10,
        3,   3]
```

> ⚠️ 官方原文：**"推荐在实时系统中按控制频率 >= 500Hz 要求来控制机械臂运动，
> 以保证控制效果和稳定性，否则可能损坏机器。"**
> 仓库的应对是把 30 Hz 的 chunk **线性插值到 500 Hz** 再逐条发
> （`servoj_frequency = 500`）。**不要按 30 Hz 直接灌 servoj。**

### 9.6 🆕 任务路由：5 个任务共用一个模型

训练了 **5 个任务**，全部由**同一个模型**承担，唯一的路由手段就是 `task_description`：

```
Press the red button
Press the black button
Press the green button
Switch the selector from left to right      ← 需要夹爪
Switch the selector from right to left      ← 需要夹爪
```

> ⚠️ **发错任务文本不会报错，只会做错事。** 客户端必须让操作员显式选择，
> 且不允许"默认值凑合"。

### 9.7 🆕 客户端安全层（v2 已实现）

在 `tron2_robot_client.py` 里，从模型输出到"可执行指令"之间加了三道：

1. **左臂锁定** —— 左臂 7 维直接替换成**实测位姿**（模型输出整个丢弃）。
   比限幅更强：它不可能动。
2. **绝对限位** —— 按 TRON2 用户手册 §1.5 的关节限位表做软限位。
   实现上取 `[min(手册下限, 当前值), max(手册上限, 当前值)]`，
   **保证不会把机器人从它当前所在的位置往外挤**。
3. **幅值抑制** —— 每个动作值相对实测位姿不超过 `--max-delta`（默认 0.10 rad≈5.7°），
   头部 0.05 rad。这是**与命名无关**的保护。

🆕 **实测效果**（拧旋钮任务，真实观测）：原始输出里有一个动作步偏离当前位姿
**0.8905 rad（51°）**（右腕），三道之后压到 **0.1000 rad**。

关于**命名歧义**：手册用 `proximal_pitch/roll/yaw/elbow`，真机 `/joint_states` 用
`abad/hip/yaw/knee`（厂家在足式命名体系下的叫法）。v2 用采集数据做了对齐检验：
`wrist_yaw_R` 实测下限 −1.394 vs 手册 −1.39、`wrist_pitch_R` 实测 −0.786 vs 手册 −0.78，
**两个维度几乎精确贴住手册边界**，说明**索引顺序与手册一致**。
（仍建议向厂家确认一次。）

### 9.8 🆕 上机步骤（渐进，每步都要人守急停）

`tron2_robot_client.py --selftest {movej,servoj,gripper}` 会**发一条内容是"当前位姿"的指令**
（期望零运动），用来验证指令通道：

| 步 | 命令 | 验证什么 | 预期 |
|---|---|---|---|
| 1 | `--selftest movej` | `request_movej` 14 维是否被接受 | 响应 success，关节不动 |
| 2 | `--selftest servoj` | `request_servoj` 16 维格式（两种说法哪个对） | 响应无 / `notify_servoJ` 失败，关节不动 |
| 3 | `--selftest gripper` | `request_set_limx_2fclaw_cmd` 是否被接受 | 夹爪不动 |
| 4 | 只执行 chunk 第 1 步，`--max-delta 0.02` | 小幅真实运动 | 动作可预期、无抖动 |
| 5 | 逐步放开 `--max-delta` 到 0.05 → 0.10 | 稳定性 | 逐级确认 |

**每一步都必须**：① 确认没人在遥操（`tele_operation` 一直报 OK，**不能**用来判断是否正在操控）；
② 有人手放在**本体硬急停**上（软急停在运动中无效，NJ-01/NR-01）。

### 9.9 🆕 端到端验证结果（只读，v2 实测）

用真机实时观测跑 `tron2_robot_client.py`：

```
观测耗时 15~29 ms     推理往返 60~63 ms（服务端 ~52 ms）     输出 (50, 18) denormalized
右臂相对当前位姿 max|Δ| = 0.016 ~ 0.025
左夹爪 idx16 = 0.0000（恒为 0，§7.2 预期）
头部 idx14-15 = [0.5661, 0.0059]  ← 精确等于当前头部位置
夹爪 hw[0, 2] → 标定后 [0.0000, 0.0200]
finite = True，机器人全程静止
```

同时验证了**冷启动修复在真机观测上成立**（§7.8）：含 Triton JIT 的第一次调用
右臂 Δ=0.0168，与后续 0.018~0.025 一致，没有出现修复前的 0.41。

---

## 附录 A：这个环境里跑过什么

### A.1 原始机器（Ubuntu 22.04）

| 操作 | 结果 |
|---|---|
| batch / 显存扫描 | batch 2 → 16480 MiB；batch 3 → 20538 MiB；batch 4 → OOM |
| 梯度检查点 ON/OFF 对比 | 都是 1.75 s/it、16480 MiB → 无差别，保持 ON |
| 冒烟测试（18 步） | 通过，无线程池错误、无 OOM |
| 检查点保存测试 | 成功产出 2 份，单份 29.2 GB |
| **正式训练** | **17000 步 / 20h27m / loss 0.212 → 0.0088 / 0 次尖峰** |
| 推理延迟 | 朴素 236.3 ms / 加速 45.8 ms（5.16×） |
| 布局修复测试 | 41 passed |

### A.2 🆕 重建机器（Ubuntu 20.04）

| 操作 | 结果 |
|---|---|
| 装机时间 | 约 1 小时（含 22.7 GB 传输、约 4 GB wheel 下载） |
| 传输速率 | pod → 本机约 17–24 MB/s |
| 数据集校验 | 2.4 G / 0 软链 / 1668 mp4 ✅ |
| 训练日志复算 | 17000 步 / 中位 0.0113 / 末段 1000 步 0.0081 / lr 2.5e-06 / 尖峰 0 ✅ |
| torch GPU 验证 | 与 v1 期望输出逐行一致 ✅ |
| **单测** | **41 passed** ✅ |
| flash-attn | 换社区 manylinux 构建后可用；GPU 实算 `(2,256,8,64)` bf16 finite ✅ |
| **首次推理** | **5.6 s**（Triton JIT + CUDA Graph） |
| **冷/热首调用对比** | 修复前冷启动第 1 次右臂 max\|Δ\| = **0.4132**（确定性错误），第 2–5 次 0.0285~0.0357；修复后第 1 次 = **0.0394** ✅ |
| **稳态推理** | **20 次中位 56.2 ms wall / 48.4 ms 服务端** |
| **输出形状** | **(50, 18)** 机器人布局，`denormalized=True` ✅ |
| **右臂 delta 还原** | 紧贴当前位姿，max|Δ| = **0.032** ✅ |
| **左夹爪** | 恒为 **0.0**（§7.2 预期） ✅ |
| **head 透传** | 送 `[0.7,-0.3]` → 输出 idx 14-15 精确一致，未污染右臂 ✅ |
| 腕部首帧（数据集） | **纯黑**（mean=0.0），跳 10 帧后 mean≈113–122 |
| **真机实时端到端** | ✅ 3 路 480×640 相机 + `/joint_states` + 夹爪 → 推理服务，**只读** |
| 真机观测耗时 | 15~29 ms（3 张 JPEG + 关节 + 夹爪） |
| 真机推理往返 | 60~63 ms（服务端 ~52 ms） |
| 真机右臂 delta | 紧贴当前位姿，max\|Δ\| = **0.016~0.025** ✅ |
| 真机 head 透传 | idx 14-15 = `[0.5661, 0.0059]`，精确等于当前头部位置 ✅ |
| 真机夹爪 | hw `[0, 2]` → 标定后 `[0.0000, 0.0200]` ✅ |
| **任务路由验证** | 同一观测换 `task_description`：按按钮 → 右夹爪 max **0.047**；拧旋钮 → **0.822** ✅ |
| **安全层效果** | 原始最大偏离 0.8905 rad → 钳制后 **0.1000 rad** ✅ |
| 腕部首帧（真机） | **不是黑帧**（mean≈128/133） |

---

## 附录 B：commit 对照

| commit | 内容 | 是否需要保留 |
|---|---|---|
| `226df18` | `ParquetDatasetV3` 支持自动统计 + 补齐位监督开关 | ✅ 训练必需 |
| `b7a0115` | TRON2 训练配置 + 修正统计量 + 文档 | ✅ 必需 |
| `d567b98` | CUDA 检查本地放宽 | ⚠️ 本机环境权宜，**不要上游提 PR**；且只在驱动报 CUDA < 12.4 时才有意义 |
| `f51b41e` | 加速推理部署配置 | ✅ 部署必需 |
| `e0eb092` | 16→18 动作布局展开 + `state_permutation` | ✅ **部署必需，缺了机器人不会动** |
| 🆕 `ff2ed5f` | 补 `themis` 段 + `triton_max_prompt_len=200` + `ori_action_dim=16` + RTC 截断 | ✅ **部署必需**（§6.2/§6.6） |
| 🆕 `7f40aa6` | 冷启动首调用前重新写输入缓冲 | ✅ **必需，否则第一条动作是错的**（§7.8） |
| 🆕 `1b726a2` `49b211f` `281ddf0` | 本文档本身 | 📄 文档 |
| 🆕 未提交 | `themis` 段 + `triton_max_prompt_len=200` + `ori_action_dim=16`（部署配置） | ✅ **部署必需**（§6.2/§6.6） |
| 🆕 未提交 | `PI05FlowMatchingRTCInference` 补 `ori_action_dim` 截断 | ✅ **部署必需**（§6.6） |

---

## 附录 C：🆕 源机器上不在 git 里的辅助脚本

这些是造数据/标定用的，**机器一释放就没了**，建议一起抢救到 `pod_scripts/`：

| 文件 | 用途 |
|---|---|
| `prepare_tron2_for_fluxvla.py` | 把原始采集数据转成 FluxVLA 的 lerobot 结构 |
| `check_lerobot_dataset.py` | 数据集完整性检查 |
| `make_tron2_lora_config.py` | 生成 TRON2 LoRA 训练配置 |
| `sweep_batch_ckpt.sh` | batch / 显存 / 检查点扫描（附录 A.1 那些数字的来源） |
| `sweep_dataloader.sh` | DataLoader worker 扫描 |
| `sweep_speed.sh` | 训练速度扫描 |

---

## 附录 D：🆕 v2 相对 v1 的修订清单

| 位置 | v1 说法 | v2 修订 |
|---|---|---|
| §3.3.1 | 基座权重"纯推理可省" | ❌ **错的**，`pretrained_name_or_path` 引用了它，推理必需 |
| §3.3.1 | （未提 `checkpoints/pi05_base`） | 需要它的 tokenizer，但**不要**拷里面 14 GB 的 fp32 权重 |
| §3.4 | （未提 GitHub HTTPS 阻断） | 新增 §3.4.1 git SSH 重写 |
| §3.4 | （未提 flash-attn glibc 问题） | 新增 §3.4.2 换 manylinux 构建 |
| §3.4 | （未提 `libcuda.so` 缺失） | 新增 §3.4.3：**优先 `sudo ln -s` 建系统级软链**，无 sudo 时才用 `TRITON_LIBCUDA_PATH` 兜底 |
| §7.8 | "首次推理要先 warmup" | 补充：warmup 不只是延迟问题，旧版本**冷启动第一条动作是错的**（§7.8） |
| §3.4 | （未提脚本会中途 abort） | 新增 §3.4.4 手动补 `pip install -e .` |
| §1.3 | env 路径被脚本硬编码 | 修正：只是 env **名字**必须叫 `fluxvla`，位置随意 |
| §1.2/§3.4 | 本地补丁的必要性 | 补充：驱动报 ≥ 12.4 时该检查本来就通过 |
| §6.1/§6.2 | 部署配置可直接启动服务 | ❌ 缺 `themis` 段，已补 |
| §6.3 | 首次推理 29.3 s | 实测 5.6 s（差异可能来自 prompt 长度与机器） |
| §6.5 | （未写观测契约） | 新增 §6.5：`qpos` 16 维 + `states` 18 维 |
| §6.6 | `triton_max_prompt_len=48` / `ori_action_dim=14` | ❌ 两个值都会崩，改为 200 / 16 |
| §7.9 | 腕部首帧黑"1~6 帧" | 实测更严重，跳 10 帧才稳 |
| §7.10 | （未提） | 新增：两套 ZMQ 实现容易混淆 |
| §9 | 网络/上机器人准备 | 新增整节 §9，含有线网络排障 |
| §7.2 | 把夹爪整体当作退化维度 | ❌ **右夹爪是有效的**：5 个任务里有 2 个拧旋钮任务需要它（gripR 跑到 0.94）。只有左臂 + **左**夹爪退化 |
| §6.5 | （未写夹爪量纲） | 补充：`/gripper_state` / `get_limx_2fclaw_state` 是 **hw 0-100**，数据集存的是 **÷100** 后的 0-1；实测 `right_opening=2` → 0.02 |
| §6.5 | 任务文本当作常量 | ❌ 5 个任务共用一个模型，**`task_description` 是唯一路由手段**，发错不报错但做错事（§9.6） |
| §9.4 | （未写相机通道顺序） | 新增：compressed 话题是 **bgr8 JPEG**，`cv2` 解出 BGR 而训练用 RGB，客户端**必须 `[..., ::-1]`** |
| §9.5 | （未写指令接口） | 新增：`movej` 14 维 / `servoj` 16 维 / `limx_2fclaw` 夹爪，含正增益与 500 Hz 插值要求；并指出**官方文档的 servoj 报文与仓库实现不一致** |
