# π0.5 + TRON2 柜内操作 — 训练配置完整说明

> 配置文件：`configs/pi05/pi05_paligemma_tron2_cabinet_lora.py`
> 归一化统计：`datasets/RealRobot_Tron2_lerobot/tron2_stats_armsymmetric.json`
> 文档更新时间：2026-09-29（含归一化修正、磁盘运维、部署链路）
> 增补：2026-10-03 —— 检查点/续训修复与 csv 训练指标，见第 13 章
> 当前状态：v2b 训练运行中（step 597 / 17000，中位 loss 0.064，无尖峰）

---

## 目录

1. [一页速览](#一页速览)
2. [硬件与软件环境](#硬件与软件环境)
3. [数据集](#数据集)
4. [模型架构](#模型架构)
5. [LoRA 微调策略](#lora-微调策略)
6. [数据变换流水线](#数据变换流水线)
7. [**归一化统计量修正（核心）**](#归一化统计量修正核心)
8. [训练超参数](#训练超参数)
9. [与厂商原版的偏离对照](#与厂商原版的偏离对照)
10. [性能与时间预算](#性能与时间预算)
11. [磁盘与运维（回收站陷阱）](#磁盘与运维回收站陷阱)
12. [启动与恢复](#启动与恢复)
13. [**检查点、续训与训练监控**](#检查点续训与训练监控)
14. [已知坑与必读注意事项](#已知坑与必读注意事项)
15. [**部署链路**](#部署链路)
16. [监控与产物](#监控与产物)
17. [训练历史与对照实验](#训练历史与对照实验)

---

## 一页速览

| 项目 | 值 |
|---|---|
| 模型 | π0.5 (PI05FlowMatching)，Gemma-2B 主干 + SigLIP-224 视觉塔 |
| 微调方式 | **LoRA**（rank=32, alpha=64），可训练参数 **1.97%**（72.7 M / 3.69 B） |
| 基座权重 | `checkpoints/pi05_base_bf16/model.safetensors`（7.23 GB, bf16） |
| 数据 | 5 个 LeRobot v3.0 数据集，**556 episodes / 56,485 帧**，5 个柜内子任务 |
| 动作空间 | **16 维** `[左臂7, 左夹爪, 右臂7, 右夹爪]`，相对动作（夹爪绝对） |
| 相机 | `cam_high` / `cam_left_wrist` / `cam_right_wrist`（224×224） |
| 归一化 | quantile，**使用人工修正后的统计量**（见第 7 章） |
| 有效 batch | **24**（micro-batch 3 × 累积 8 × 单卡） |
| 训练步数 | **17,000 步**（≈ 7.22 epoch，2,354 步/epoch） |
| 优化器 / LR | AdamW，峰值 lr **2.5e-5**，warmup 1000 + cosine 退火到 2.5e-6 |
| 精度 | bf16 计算 + **bf16 主权重**（`keep_params_fp32=False`） |
| 梯度检查点 | 开启（注入 GemmaAttention + Block） |
| 检查点 | 每 1000 步滚动保存，保留最近 3 份，每份 **29.2 GB** |
| 预计耗时 | **≈ 20 小时**（稳态 4.21 s/步 + 17 次保存各 32 s） |
| 峰值显存 | **20,534 MiB / 24,564 MiB**（余量 ≈ 4 GB） |

---

## 硬件与软件环境

### 硬件

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D |
| 显存 | 24,564 MiB（约 24 GB） |
| 计算能力 | sm_89（Ada Lovelace） |
| GPU 数量 | **1**（`Distributed World Size = 1`） |
| CPU | 44 核 |
| 内存 | 78 GB |
| 根分区 | 196 GB（`overlay`，`upperdir=/ebs/upper`，底层 `/dev/vdd`） |

### 运行时

| 组件 | 版本 |
|---|---|
| Python | 3.10.21（conda 环境 `/opt/miniconda3/envs/fluxvla`） |
| PyTorch | 2.6.0+cu124 |
| CUDA 运行时 | 12.4 |
| CUDA 工具包 | 12.4（`/usr/local/cuda`） |
| cuDNN | 90100 |
| flash-attn | 2.8.3（源码编译，仅 sm_89） |
| transformers | 5.3.0 |
| pydantic | 2.13.4 |
| numpy / pandas / pyarrow | 1.26.4 / 2.3.3 / 24.0.0 |

### 驱动（重要）

```
nvidia-smi 报告     : 535.154.05
内核模块 (/proc/driver/nvidia/version)
                    : NVIDIA UNIX x86_64 Kernel Module 535.154.05
运行环境            : K8s Pod（/proc/1/cgroup → /kubepods/.../user-container）
```

**驱动在宿主机内核里，Pod 内不能也不应修改。**
CUDA 12.4 的负载能跑在 535 驱动上，靠的是 CUDA 的 minor version compatibility（同一大版本内向后兼容），实测矩阵乘、bf16 均正常。

### flash-attn 编译说明

官方 PyPI/GitHub 的 flash-attn wheel 用新 C++ ABI（`cxx11abiTRUE`）编译，与本环境的 torch（`cxx11abiFALSE`）不匹配，导入时报：

```
undefined symbol: c10::Error::Error(..., std::__cxx11::basic_string...)
```

**解法**：从源码编译，且只编本卡架构，省掉 3/4 工作量：

```bash
unset PYTHONPATH
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=16
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # 跳过从 GitHub 下预编译 wheel（国内极慢）
export FLASH_ATTN_CUDA_ARCHS=89           # 只编 sm_89（4090）
pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

> 副作用：产出的 wheel 只能在 sm_89 卡上使用。将来换 A100/H100 需重编。

### 无害警告（可忽略）

| 警告 | 判定 | 依据 |
|---|---|---|
| `UnsupportedFieldAttributeWarning`（pydantic 2.13.4） | 忽略 | 框架按旧版 pydantic 写的；仅影响 `repr`/`frozen` 元数据语义，不在训练路径上 |
| 分词器 `incorrect regex pattern`（transformers 5.3.0） | **误报** | 该分词器 `pre_tokenizer` 是朴素 `Split(" ")` 不含正则；编码/解码往返（含数字 `'number 1234 here'`）完全正确；且 VLA 只 encode 不 decode。且此版本的 `fix_mistral_regex=True` 会抛 `TypeError` |
| `'llm_expert.embed_tokens.weight' not found... skipping` | 忽略 | 动作专家本就不需要独立词嵌入 |

**日志实测**：68 行日志中 ERROR / Traceback 为 **0 条**。

若要彻底静音（需重启生效）：

```bash
TRANSFORMERS_VERBOSITY=error      # 抑制分词器警告
PYTHONWARNINGS=ignore             # 抑制 pydantic 警告
```

---

## 数据集

### 来源与规模

原始导出：`/root/tron_ws/FluxVLA/datasets/lerobot_*.tar.gz`（5 个包，2.4 GB）
解包后：`/root/tron_ws/datasets_raw/`
转换后（训练实际读取）：`/root/tron_ws/FluxVLA/datasets/RealRobot_Tron2_lerobot/`

| 数据集目录 | 任务文本 | episodes | 转换后帧数 |
|---|---|---|---|
| `lerobot_2026-09-28_22-05-37` | `Press the red button` | 102 | 8,834 |
| `lerobot_2026-09-28_22-26-43` | `Press the black button` | 104 | 8,242 |
| `lerobot_2026-09-28_22-36-10` | `Press the green button` | 107 | 8,097 |
| `lerobot_2026-09-28_22-44-32` | `Switch the selector from left to right` | 134 | 17,875 |
| `lerobot_2026-09-28_22-54-05` | `Switch the selector from right to left` | 109 | 13,437 |
| **合计** | — | **556** | **56,485** |

- 格式：**LeRobot v3.0**（`codebase_version: v3.0`）
- `robot_type`：`tron2_v4_claw_v0`
- 帧率：**30 fps**（由 episode 0 = 120 帧 / 4.0 秒推得）
- 单帧数据文件：`data/chunk-XXX/file-XXX.parquet`（v3.0 按容量切分，episode 用 `episode_index` 列区分）

### 维度裁剪：21 → 16

数采软件导出 **21 维**，FluxVLA 的 TRON2 配置期望 **16 维**。

| idx | 名称 | 归属 | 是否保留 |
|---|---|---|---|
| 0–6 | `abad/hip/yaw/knee/wrist_yaw/wrist_pitch/wrist_roll_L_Joint` | 左臂 7 关节 | ✅ |
| 7 | `left_gripper` | 左夹爪 | ✅ |
| 8–14 | 同名 `_R_Joint` × 7 | 右臂 7 关节 | ✅ |
| 15 | `right_gripper` | 右夹爪 | ✅ |
| 16–17 | `head_pitch_Joint` / `head_yaw_Joint` | 头部 | ❌ 丢弃 |
| 18–19 | `linear_x` / `angular_z` | 底盘 | ❌ 丢弃 |
| 20 | `lifter_joint` | 升降柱 | ❌ 丢弃 |

**裁剪依据（均有数据支撑）**：

- 底盘三维在全部帧中**非零帧数 = 0**（`linear_x`/`angular_z`/`lifter` 全程恒为 0）
- 头部两维在全部 56,888 帧中**只有 6~7 个不同取值**（`head_pitch` std=0.0024、`head_yaw` std=0.00005），且部署时 `enable_head_control` 默认 False → 头部指令根本不发送
- **不需要重排**：`tron2_operator.py` 的 `joint_names` 顺序与数据天然一致（左臂 7 → 右臂 7 → 头 2）

**注意**：`NormalizeStatesAndActions` 只会把 16 维**补零到 32 维**，不会截断。所以 21→16 必须在**数据侧**做，不能指望 transform 兜底。

### 数据质量发现

| 发现 | 严重度 | 处理 |
|---|---|---|
| 任务文本含**零宽空格 U+200B**（`\u200bPress the black button\u200b`） | 中 | 转换脚本已清洗 |
| `tasks.parquet` 列名错位（文本被塞进 `__index_level_0__`） | 低 | FluxVLA 已兼容该格式 |
| **左臂全程静止**（帧间步进 0.00002，右臂 0.00605） | **高** | 引发归一化 span 塌缩，见第 7 章 —— 这是本项目的核心问题 |
| **左夹爪(idx 7) 在全部 5 个数据集恒为 0** | 中 | 接受：纯右手操作任务 |
| 三个"按按钮"数据集**右夹爪也恒为 0** | 低 | 合理：按按钮不需要夹爪开合 |
| 头部两维几乎不动（仅 6~7 个取值） | 中 | 已丢弃；若将来要用头部需重采 |
| **腕部相机约 30% episode 前 1–6 帧纯黑** | **高** | 需处理，见下 |
| 第 0 帧关节状态全零（每 episode 起始） | 低 | 转换脚本已丢弃（总计 -403 帧） |

**腕部相机黑帧根因**：头部相机持续推流，腕部相机在 episode 开始时才启动，有 1–6 帧（33–200 ms）启动延迟。与"第 0 帧关节状态全零"同源 —— **录制在传感器就绪前就开始了**。

| 相机 | 视频数 | 首帧全黑 | 占比 |
|---|---|---|---|
| `cam_high` | 556 | 0 | **0.0%** |
| `cam_left_wrist` | 556 | 179 | **32.2%** |
| `cam_right_wrist` | 556 | 153 | **27.5%** |

**为什么这条最要命**：柜内操作必须靠腕部相机（头部会被门框挡死、柜内又暗）。模型最依赖的输入，恰好在约 30% 的 episode 开头是纯黑，会教出"黑图 → 某动作"的脏关联。**建议后续重采时先启动相机再开始录制。**

---

## 模型架构

### 主干

```python
type = 'PI05FlowMatching'
```

| 组件 | 关键参数 |
|---|---|
| **LLM 主干** (`llm_backbone`) | `ConditionGemmaModel`，hidden=2048，layers=18，heads=8，KV heads=1，head_dim=256，intermediate=16384，vocab=257152，rope_theta=10000 |
| **视觉塔** (`vision_backbone`) | `SigLIPViTBackbone`（`siglip_224`），hidden=1152，layers=27，heads=16，patch=14，image=224，intermediate=4304 |
| **投影层** (`projector`) | `LinearProjector` 1152 → 2048 |
| **动作专家** (`llm_expert`) | `ConditionGemmaModel`，hidden=1024，layers=18，heads=8，KV heads=1，intermediate=4096，**use_adarms=True**，adarms_cond_dim=1024 |

### 动作头与流匹配

| 参数 | 值 | 说明 |
|---|---|---|
| `proj_width` | 1024 | 动作专家宽度 |
| `n_action_steps` | 50 | 单次预测的动作序列长度 |
| `max_action_dim` | 32 | 模型内部动作维度（数据 16 维补零到此） |
| `action_in_proj` | 32 → 1024 | |
| `action_out_proj` | 1024 → 32 | |
| `time_mlp_in` / `time_mlp_out` | 1024 → 1024 | 流匹配时间步嵌入 |
| `time_sampler` | `beta` | 时间步采样分布（α=1.5, β=1.0） |
| **`openpi_fp32_flow`** | **`False`** | 流匹配用 bf16 而非 fp32（**与厂商不同**） |
| `ori_action_dim` | 16 | 真实数据维度 |
| **`loss_action_dim`** | **16** | 只监督前 16 维（**与厂商不同**，厂商为 32） |

### 参数规模（实测）

```
all params       : 3,689,500,464   (3.69 B)
trainable params :    72,742,944   (72.7 M)
trainable %      :          1.97 %
```

### 权重加载映射

基座：`./checkpoints/pi05_base_bf16/model.safetensors`（**7.23 GB，bf16 版**）

```python
name_mapping = {
    'llm_backbone':            'paligemma_with_expert.paligemma.model.language_model',
    'vision_backbone.vision':  'paligemma_with_expert.paligemma.model.vision_tower',
    'projector.projector':     '...multi_modal_projector.linear',
    'llm_expert':              'paligemma_with_expert.gemma_expert.model',
    'time_mlp_in.projector':   'time_mlp_in',
    'time_mlp_out.projector':  'time_mlp_out',
    'action_in_proj.projector':  'action_in_proj',
    'action_out_proj.projector': 'action_out_proj',
    'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
}
```

---

## LoRA 微调策略

**为什么用 LoRA 而不是全量微调** —— 单张 24 GB 卡装不下全量微调：

```
fp32 master 参数   14.5 GB      (3.69 B × 4 B)
bf16 计算副本       7.2 GB
AdamW 两个动量态    29.0 GB      (fp32)
梯度                7.0 GB
────────────────────────────────
合计               ~58 GB   >   24 GB    ❌ 必然 OOM
```

### LoRA 参数

| 参数 | 值 |
|---|---|
| `use_lora` | `True` |
| `lora_rank` | **32** |
| `lora_alpha` | **64**（等效缩放 α/r = 2.0） |
| `lora_dropout` | 0.0 |

### LoRA 注入目标（11 组模块）

```python
lora_target_modules = [
    'q_proj', 'v_proj', 'k_proj', 'o_proj',      # 注意力
    'gate_proj', 'up_proj', 'down_proj',          # MLP
    'projector.projector',                        # 视觉→语言投影
    'out_proj', 'fc1', 'fc2',                     # 视觉塔内部
]
```

### 额外可训练模块（`modules_to_save`）

动作头相关的 4 个投影层**完整训练**（不是低秩）：

```python
modules_to_save = [
    'action_in_proj', 'action_out_proj',
    'time_mlp_in', 'time_mlp_out',
]
```

`freeze_llm_backbone` 与 `freeze_vision_backbone` 均为 `False`（冻结由 LoRA 机制本身实现，而非这两个开关）。

---

## 数据变换流水线

`train_dataloader.dataset` 是 `DistributedRepeatingDataset` 包装 `ParquetDatasetV3`。

### 外层（DistributedRepeatingDataset）

```python
dataset_statistics_path = './datasets/RealRobot_Tron2_lerobot/tron2_stats_armsymmetric.json'
name_mappings           = {'observation.state': ['proprio'], 'action': ['action']}
statistic_keys          = ['observation.state', 'action']
```

> **不再使用 `auto_compute_statistics`**，改为手工提供修正后的统计文件（原因见第 7 章）。
>
> 优先级（`scripts/train.py:389-397`）：`dataset_statistics`（内联） > `dataset_statistics_path` > `auto_compute_statistics`。
>
> 关键细节：**提供 stats 文件后，`name_mappings` 的重命名步骤会被跳过** —— 所以文件里的键名必须已经是 `proprio` / `action`（自动生成的版本正好如此）。

### 内层转换链（按执行顺序）

| # | Transform | 作用 |
|---|---|---|
| 1 | `ProcessParquetInputs` | 读 parquet 列 + 解码三路视频。**视频路径拼装就在这里** |
| 2 | `RelativeActions` | 前 16 维转为相对动作：`mask=[True]*7+[False]+[True]*7+[False]` |
| 3 | `NormalizeStatesAndActions` | quantile 归一化，`state_dim=32, action_dim=32`（**只补齐不截断**） |
| 4 | `PreparePromptWithState` | 组装 prompt |
| 5 | `ProcessPrompts` | 分词，`max_len=200`，tokenizer 来自 `checkpoints/pi05_base` |
| 6 | `ResizeImagesWithPad` | 缩放到 224×224 |
| 7 | `SimpleNormalizeImages` | 图像归一化 |
| 8 | `OpenPIImageAugment` | 图像增强，`base_camera_indices=(0,)` 只增强头部相机 |

**`RelativeActions` 的确切语义（容易误解）**：

```183:186:fluxvla/transforms/transform_actions.py
dims = self.mask.shape[-1]
actions[..., :dims] -= np.expand_dims(
    np.where(self.mask, states[..., :dims], 0), axis=-2)
```

`axis=-2` 展开成 `(1, dims)` 对 `(T, dims)` 广播 → **窗口内 50 步的动作全都减同一个 state（窗口起始帧的那个）**，而不是逐步 `action[t] - state[t]`。这是 OpenPI 的约定（"相对当前位姿的增量"）。

### 数据集级参数

```python
action_key                 = 'action'
window_start_idx           = 0
action_window_size         = 50     # 每次取 50 步动作窗
supervise_terminal_padding = True   # 末端 padding 也纳入监督
```

> `supervise_terminal_padding` 是**为支持 v3.0 数据集而打的补丁**（见 `parquet_dataset_v3.py`）。

---

## 归一化统计量修正（核心）

### 问题：quantile 归一化的分母塌缩

归一化公式（`fluxvla/transforms/normalize.py:773-777`）：

```python
def _normalize_quantile(self, x, stats):
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

它把 `[q01, q99]` 映射到 `[-1, +1]` 且**不裁剪**。所以 `q99 - q01` 越小，放大倍数越大。

**我们的数据显示左臂 7 维的分母塌缩到了 0.005~0.008：**

| 维度 | 原始 span (q99−q01) | 放大倍数 |
|---|---|---|
| `abad_L` | 0.00767 | 130× |
| `hip_L` | 0.00748 | 134× |
| `yaw_L` | 0.00516 | **194×** |
| `knee_L` | 0.00670 | 149× |
| `wyaw_L` | 0.00535 | 187× |
| `wpitch_L` | 0.00629 | 159× |
| `wroll_L` | 0.00571 | 175× |
| `grip_L` | 0.00000 | 1e9×（值恒为 0，实际无害） |
| **右臂对照（正常）** | 0.51 ~ 1.95 | 0.51 ~ 1.30× |

**根因**：左臂全程静止 → 相对动作几乎恒为 0 → 98% 的分布挤在 ±0.006（纯噪声），只有 2% 是真运动（可达 ±1.5）。quantile 只圈住了噪声部分，于是：

- 静止噪声被放大到 **O(1)**
- 偶发的真实小动作（0.18 rad）被放大到 **54** → MSE 贡献 **2916**

### 量化影响（实测，50,322 个窗口，两种独立口径）

**目标方差占比（占 16 维总方差）**：

| | 左臂 | 右臂 | 夹爪 |
|---|---|---|---|
| 修正前 | **98.8%** | 0.4% | 0.8% |
| 修正后 | **2.3%** | 35.1% | 62.5% |

**极端值窗口占比**：

| 口径 | >10 | >20 | >50 | >100 |
|---|---|---|---|---|
| 逐步 per-timestep 修正前 | 35.30% | 19.82% | 2.57% | 0.00% |
| 逐步 per-timestep 修正后 | 0.00% | 0.00% | 0.00% | 0.00% |
| 窗口均值 修正前 | 18.17% | 6.14% | 0.75% | 0.00% |
| 窗口均值 修正后 | 0.00% | 0.00% | 0.00% | 0.00% |

**最坏窗口**：

```
逐步口径   修正前: max=461.7  p99=71.9  p50=1.22
           修正后: max=  2.4  p99= 1.9  p50=1.00
```

**目标均方**：

```
窗口均值口径   12.13 -> 0.153   降低 79.5 倍
逐步口径       18.18 -> 0.192   降低 94.8 倍
```

**实测最大 loss 尖峰** 629（step 1195，同期基线 0.05）。

> **两处数据修正记录**（早期测量有误，已废弃）：
>
> 1. 曾表述"左臂占 **67.5%** 目标方差"→ 正确值是 **98.8%**（早期脚本统计口径不全）
> 2. 曾表述"\|归一化值\|>10 的窗口只有 **27 个 / 0.091%**"→ 正确值是 **18.17%**（早期脚本只遍历了部分窗口）
> 3. 曾用"某个 0.18 rad 例子下 loss 从 2916 降到 0.055（53000 倍）"来表述效果 —— 那是单点极值，**聚合口径是降低 79.5~94.8 倍**，应以聚合值为准
>
> **"目标方差占比"也不等于"可达到 loss 占比"** —— 其中每段 episode 固定的静止位姿偏置是可预测的。真正的硬指标是**极端值窗口归零**和**收敛速度**。

### 待优化：夹爪占了修正后方差的 62.5%

`grip_L` 原始值恒为 0，而 `q01 = q99 = 0`：

```
(0-0)/(0+1e-6)*2-1 = -1     ← 常数目标 -1
```

模型要花 1/16 的 loss 去学这个**无信息的常数**。修法：把 `q01/q99` 设为 `-1/1`，使目标正好落在 0：

```
(0-(-1))/(1-(-1))*2-1 = 0
```

这是瞬态开销（常数收敛很快），**不值得为此重启训练**，记录待下一轮处理。

### 修法

**左臂 7 维（idx 0–6）改用对称右臂（idx 8–14）的 `q01/q99`。**

物理依据：同型号机械臂的同名关节具有相同的运动范围；左臂因静止无法估出有意义的范围，用对称关节替代是合理的。`proprio` 与 `action` 两个键都做同样处理。

生成脚本（已执行，产物已落盘）：

```python
PAIR = [(i, i+8) for i in range(7)]      # 左臂 0-6  <->  右臂 8-14
fixed = copy.deepcopy(auto_stats)
for key in ('action', 'proprio'):
    d = fixed['private'][key]
    for i, j in PAIR:
        d['q01'][i] = d['q01'][j]
        d['q99'][i] = d['q99'][j]
```

### 修正效果

| 维度 | 修正前 span | 修正后 span | 放大倍数变化 |
|---|---|---|---|
| `abad_L` | 0.00767 | 1.05081 | 130× → **0.95×** |
| `hip_L` | 0.00748 | 0.76973 | 134× → **1.30×** |
| `yaw_L` | 0.00516 | 1.03617 | 194× → **0.97×** |
| `knee_L` | 0.00670 | 1.64347 | 149× → **0.61×** |
| `wyaw_L` | 0.00535 | 1.95058 | 187× → **0.51×** |
| `wpitch_L` | 0.00629 | 0.88522 | 159× → **1.13×** |
| `wroll_L` | 0.00571 | 0.78888 | 175× → **1.27×** |

| 场景 | 修正前 | 修正后 |
|---|---|---|
| 左臂静止噪声 | 归一化 ≈ 1.0 | **0.0002** |
| 左臂小动作 0.18 rad | 归一化 **54.0**，loss 2916 | 归一化 **0.234**，**loss 0.055** |

### 为什么不扩到 18 维（把头部纳入）

曾考虑把动作空间改为 18 维 `[L7, R7, head2, gripL, gripR]` 以匹配机器人原生顺序。**实测否决**：

```
维度           span=q99-q01    放大倍数
head_pitch       0.00060       1667 倍    ← 比左臂还糟 10 倍
head_yaw         0.00040       2500 倍
```

头部两维在全部 56,888 帧中**只有 6~7 个不同取值**（`head_pitch` std=0.0024、`head_yaw` std=0.00005）。扩到 18 维等于把刚修好的问题以 10 倍强度重现，而且 `enable_head_control` 默认 False → 头部指令在部署时根本不发送，学了也没用。

**结论：保持 16 维，布局差异在部署侧用适配层解决（见第 14 章）。**

### 教训

**先查数据分布，再决定布局。** 布局一致性是必要条件，但数据能不能支撑那个布局才是决定性的。

---

## 训练超参数

```python
runner = dict(
    type='DDPTrainRunner',          # 单卡用 DDP，非 FSDP
    max_epochs=None,                # 用 max_steps 控制
    max_steps=17000,                # 56485 x 7.22 epoch / 24
    save_iter_interval=1000,        # 每 1000 步滚动保存
    max_keep_ckpts=3,               # 保留最近 3 份
    grad_accumulation_steps=8,
    seed=42,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    reduce_in_full_precision=True,
    sampler=None,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        warmup_steps=1000,
        decay_steps=17000,          # = max_steps，完整退火
        min_lr=2.5e-6,
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs',
        window_size=1,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=False,         # 主权重保持 bf16，省一半显存
    static_graph=False,
)
```

### 派生量

| 量 | 计算 | 值 |
|---|---|---|
| 有效 batch | 3 × 1 GPU × 8 | **24** |
| 步/epoch | 56,485 / 24 | **2,354** |
| 总 epoch | 17,000 / 2,354 | **≈ 7.22** |
| 总样本数 | 17,000 × 24 | **408,000** |
| warmup 占比 | 1000 / 17000 | 5.9% |
| 保存次数 | 17,000 / 1000 | **17 次** |

### LR 轨迹

```
阶段 1  线性 warmup   step 0 → 1000      lr: 0 → 2.5e-5     （上升）
阶段 2  余弦退火      step 1000 → 17000   lr: 2.5e-5 → 2.5e-6  （下降）
```

| step | lr | 累计时间 |
|---|---|---|
| 250 | 6.25e-6 | ~0.3 h |
| 500 | 1.25e-5 | ~0.6 h |
| **1000** | **2.50e-5（峰值）** | **~1.2 h** |
| 2000 | 2.48e-5 | ~2.4 h |
| 8000 | 1.59e-5 | ~9.4 h |
| 17000 | 2.50e-6 | ~20 h |

### Collator

```python
collator = dict(
    type='DictCollator',
    keys=['states', 'timestamp', 'images', 'img_masks',
          'lang_tokens', 'lang_masks', 'actions', 'action_masks'],
    meta_keys=['task_description', 'prompt', 'info', 'stats'],
)
```

### 梯度检查点

开启后实际注入到：

```
['GemmaAttention', 'Block']
```

**实测开启与关闭的显存和速度完全相同**（1.75 s/it、16,480 MiB）—— 零成本的纯保险，保持开启。

---

## 与厂商原版的偏离对照

厂商参考配置：`configs/pi05/pi05_paligemma_tron2_full_finetune.py`（设计给 4~8×GPU，**全量微调**）

| 参数 | 厂商原版 | 本配置 | 偏离理由 |
|---|---|---|---|
| 微调方式 | 全量微调 | **LoRA** | 全量微调单卡需 ~58 GB，24 GB 装不下 |
| runner | `FSDPTrainRunner` | **`DDPTrainRunner`** | 单卡无需分片 |
| `keep_params_fp32` | `True` | **`False`** | 省一半显存 |
| 基座权重 | `pi05_base`（fp32, 14.5 GB） | **`pi05_base_bf16`（7.23 GB）** | 与 bf16 训练匹配 |
| `per_device_batch_size` | 8 | **3** | 单卡 micro-batch 上限（4 会 OOM） |
| `grad_accumulation_steps` | 2 (× 4 GPU = 64) | **8** | 有效 batch 24 |
| `max_steps` | 20,000 | **17,000** | 按 5 个数据集 7.22 epoch 折算，且需被 `save_iter_interval` 整除 |
| `lr` | 2.5e-5 | **2.5e-5（不变）** | 见下方说明 |
| `decay_steps` | 30,000（> max_steps，半程退火） | **17,000（= max_steps，完整退火）** | 完整退火对最终权重更友好 |
| `loss_action_dim` | 32（监督补齐维） | **16** | 只监督真实 16 维 |
| `openpi_fp32_flow` | `True` | **`False`** | bf16 流匹配，省显存提速 |
| 数据类 | `ParquetDataset`（v2.1） | **`ParquetDatasetV3`** | 数采软件只能导出 v3.0 |
| 统计量 | `auto_compute_statistics` | **人工修正的 stats 文件** | 左臂 span 塌缩，见第 7 章 |
| wandb | 启用 | **仅 jsonl** | 本 Pod 无 API key，`wandb.init()` 会挂起 |

### 关于学习率为什么不按线性缩放下调

原本考虑按线性缩放把 lr 从 2.5e-5 降到 1e-5（× 24/64）。**核对后决定保持 2.5e-5**，理由：

1. **LoRA 的 LR 尺度与全量微调不同**：LoRA 适配器从零初始化，典型 lr 在 1e-4 ~ 3e-4 量级。本配置的 2.5e-5 已比常规 LoRA 配方低 4~10 倍，再降到 1e-5 有欠拟合风险。
2. **有效 batch 从 8 提到 24 已经让"每样本学习量"降低 3 倍**。若同时把 lr 降 2.5 倍，单 epoch 学习量会再降，需要更多 epoch 才能补回，而时间预算固定。
3. **单一变量原则**：现有 2.5e-5 + 有效 batch 8 的组合已实测收敛良好。只改 batch 能保持结果可解释。

### 关于 `loss_action_dim`

厂商设 32 并注释 "Supervise all padded model dimensions, as in OpenPI"，本配置改为 16。理由：我们的数据只有 16 维真实信息，dims 16–31 是 `NormalizeStatesAndActions` 补的常数 0，监督它们只会让模型花容量去拟合常值。改 16 后梯度信号全部落在有意义的维度上。

---

## 性能与时间预算

### 显存与吞吐（实测）

| micro-batch | 峰值显存 | 每样本耗时 | 是否有余量 |
|---|---|---|---|
| 1 | — | 0.340 s | 充足，但 GPU 利用率低 |
| 2 | 16,480 MiB | 0.219 s | 8 GB |
| **3** | **20,538 MiB** | **0.179 s** | **4 GB ✅ 采用** |
| 4 | 24,206 MiB | — | ❌ OOM（开梯度检查点也 OOM） |

**选 micro-batch 3 的理由**：显存余量 4 GB 够用（序列长度固定，显存占用确定不会随时间增长），而每样本吞吐比 batch 2 快 18%、比 batch 1 快 **1.9 倍**。

### 时间预算（实测折算）

```
稳态步时中位数        4.21 s/步
每步样本数            24
单步 = 24 × 0.179     4.30 s   ← 与实测一致，验证吞吐模型正确

训练 = 17,000 × 4.21 s ≈ 71,600 s ≈ 19.9 小时
保存 = 17 次 × 32 s    ≈ 544 s    ≈ 0.15 小时
────────────────────────────────────────────────
合计                  ≈ 20.1 小时
```

分阶段：

| 阶段 | 耗时 |
|---|---|
| 模型加载（7.23 GB safetensors） | ~50 s |
| 统计量（使用已有文件，不重算） | ~0 s |
| 首步编译开销 | ~62 s |
| **正式训练 17,000 步** | **≈ 20 小时** |

### 检查点开销（实测）

```
单份检查点 = .pt 14.75 GB + .safetensors 14.47 GB = 29.2 GB
本地保存耗时 = 32 秒（.pt 10:46:03 → .safetensors 10:46:35）
```

| 存储位置 | 写 `.safetensors` 速度 | 单次保存耗时 |
|---|---|---|
| **本地 overlay** | **~450 MB/s** | **32 秒 ✅ 采用** |
| s3fs `/personal` | 43 MB/s | ~337 秒（慢 10 倍） |

> LoRA 的检查点保存很昂贵：`ddp_train_runner.py:358-374` 会**重建一个完整基座模型 → 重新加载 7.23 GB 权重 → 合并 LoRA → 写出完整模型**，且 `.pt` 与 `.safetensors` 内容重复。

### 磁盘预算

```
每份检查点           29.2 GB
保留 3 份（滚动）     稳态 87.6 GB
保存瞬间峰值         4 × 29.2 = 116.8 GB   ← 先写新的、再删旧的
根分区可用           约 140 GB
峰值后剩余           约 32 GB   ✅ 安全
```

---

## 磁盘与运维（回收站陷阱）

### 现象

清理检查点后 `df` 显示空间**没有释放**，连续等待 100 秒也无变化，`df` 静默对照确认不是主机干扰。

### 真相

```
$ type -a rm
rm is a function          ← shell 里的 rm 被重定义了！

$ ls -la /bin/rm
-rwxr-xr-x 59912 Feb  8  2024 /bin/rm    ← 真 rm
```

**IDE 扩展（`tencent-cloud.coding-copilot`）注入了一个 `rm` shell 函数，它不删除文件，而是移到 `~/.local/share/Trash`。**

实测证据：

```
起始        avail = 53,746,588
写 200MB    avail = 53,541,788   (-200MB ✓)
/bin/rm 后  avail = 53,746,580   ← 空间回来了 ✓✓
```

清空前 `~/.local/share/Trash` 占用 **88 GB**（含我误删的 3 份旧检查点 + 测试文件）。

### 第二层陷阱：Python 层的 `os.remove` 也被钩住

```
$ tr '\0' '\n' < /proc/<train_pid>/environ | grep PYTHONPATH
PYTHONPATH=/root/.vscode-server/extensions/tencent-cloud.coding-copilot-*/out/vendor/shim
```

该目录下的 `sitecustomize.py`（52 KB）钩住了 `os.remove`（内部 `_safe_remove` → `_try_trash`）。

**后果**：训练进程的 `_cleanup_old_checkpoints()` 用 `os.remove` 删旧检查点 → 全进回收站 → 空间不释放 → 每次保存泄漏 29 GB → **约 6 次保存后磁盘耗尽**。

### 对策（必须同时做两件事）

1. **启动训练时要 `unset PYTHONPATH`**，让 shim 不加载：

```bash
setsid bash -c 'unset PYTHONPATH; export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false; \
  cd /root/tron_ws/FluxVLA; exec /opt/miniconda3/envs/fluxvla/bin/torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py --config <CONFIG> --work-dir <WORK_DIR>' \
  > <WORK_DIR>/train.log 2>&1 < /dev/null &
```

验证方法：

```bash
for p in $(pgrep -f "scripts/train.py" | head -2); do
  tr '\0' '\n' < /proc/$p/environ | grep '^PYTHONPATH' || echo "未设置 ✓"
done
```

2. **手工清理一律用 `/bin/rm`**（或 `\rm`），不要用裸 `rm`。

### 检查

```bash
# 回收站占用
du -sh /root/.local/share/Trash
# 根分区真实大文件（-xdev 只看本设备）
find / -xdev -type f -size +1G 2>/dev/null | while read f; do
  printf "%-62s %s\n" "$f" "$(du -h $f | cut -f1)"; done
# 定期清空（永久删除，谨慎）
/bin/rm -rf /root/.local/share/Trash/files/* /root/.local/share/Trash/info/*
```

### s3fs 备选方案（已评估，未采用）

`/personal` 与 `/root/Assets` 是 s3fs 挂载（16 EB 容量）。实测 5 项关键操作全部通过：

| 操作 | 结果 |
|---|---|
| `os.makedirs` | ✓ |
| `os.symlink` | ✓（`islink=True`，`readlink` 可解析） |
| `torch.save` 1 GB | ✓ 316 MB/s |
| `safetensors.save_file` 256 MB | ✓ 43 MB/s |
| `os.listdir` + `os.path.getmtime` | ✓ |
| 写入是否偷占本地磁盘 | 否 ✓ |

**但不采用**，因为本地快 10 倍（32 s vs 337 s），17 次保存能省约 85 分钟。清空回收站后本地空间充足。

---

## 启动与恢复

### 必须用 torchrun 启动（关键）

**直接 `python scripts/train.py` 会崩溃**：

```
File "fluxvla/engines/runners/fsdp_train_runner.py", line 117
    device_id = overwatch.local_rank()
AttributeError: 'PureOverwatch' object has no attribute 'local_rank'
```

原因：`overwatch` 需要 torchrun 初始化分布式上下文；纯 Python 启动得到的是 `PureOverwatch`，没有 `local_rank()`。

### 正式训练命令（完整、含所有必要环境处理）

```bash
cd /root/tron_ws/FluxVLA
mkdir -p work_dirs/tron2_cabinet_lora_v2

setsid bash -c 'unset PYTHONPATH; \
  export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error; \
  cd /root/tron_ws/FluxVLA; \
  exec /opt/miniconda3/envs/fluxvla/bin/torchrun \
    --standalone --nnodes 1 --nproc-per-node 1 \
    scripts/train.py \
    --config configs/pi05/pi05_paligemma_tron2_cabinet_lora.py \
    --work-dir work_dirs/tron2_cabinet_lora_v2' \
  > work_dirs/tron2_cabinet_lora_v2/train.log 2>&1 < /dev/null &
```

三个环境变量的必要性：

| 变量 | 为什么 |
|---|---|
| `unset PYTHONPATH` | 跳过 IDE 的 `os.remove` shim，否则检查点滚动删除会进回收站（见第 11 章） |
| `WANDB_MODE=disabled` | 本 Pod 无 `WANDB_API_KEY`，`wandb.init()` 会挂起 |
| `TRANSFORMERS_VERBOSITY=error` | 抑制分词器误报警告（可选） |

> 也可用封装好的 `scripts/train.sh <config> <work_dir>`，它会自动探测 torchrun / 火山平台的分布式环境变量 —— **但它不 unset PYTHONPATH，需自行处理**。

### 冒烟测试（改配置后建议先跑）

```bash
cd /root/tron_ws/FluxVLA
setsid bash -c 'unset PYTHONPATH; export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false; \
  cd /root/tron_ws/FluxVLA; exec /opt/miniconda3/envs/fluxvla/bin/torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train.py --config configs/pi05/pi05_paligemma_tron2_cabinet_lora.py \
  --work-dir /tmp/smoke --cfg-options runner.max_steps=40' > /tmp/smoke.log 2>&1 < /dev/null &
```

### 等待任务完成（不要用 sleep 干等）

`execute_command` 是同步的 —— 前台跑完直接返回结果。任务已转后台时，用**条件等待**（返回时刻即完成时刻）：

```bash
i=0; while pgrep -f "train.py --config configs/pi05/pi05_paligemma_tron2_cabinet_lora.py" \
        >/dev/null 2>&1 && [ $i -lt 360 ]; do sleep 5; i=$((i+1)); done
<打印结果>
```

### 断点恢复

```bash
... scripts/train.py \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora.py \
  --work-dir work_dirs/tron2_cabinet_lora_v2 \
  --resume-from work_dirs/tron2_cabinet_lora_v2/checkpoints/step-004000-epoch-001-loss=X.XXXX.pt
```

> **注意**：`--resume-from` 要用 `.pt`（含优化器/调度器状态），不是 `.safetensors`。恢复时 `max_steps` 需相应调整。

### 关键 CLI 参数

| 参数 | 说明 |
|---|---|
| `--config` | 配置文件路径（必需） |
| `--work-dir` | 日志与检查点目录 |
| `--cfg-options key=value ...` | 覆盖配置，如 `runner.max_steps=40` |
| `--resume-from` | 从检查点恢复 |
| `--eval-after-train` | 训练后自动评估 |

---

## 检查点、续训与训练监控

> 本节记录 **2026-10-03** 的排查与修复。**在此日期之前，LoRA 训练的 `--resume-from` 是不可用的**——它会静默丢弃全部 LoRA 权重（见下文）。

### 存档产物

每 `save_iter_interval` 步保存一次，在 `work_dirs/<run>/` 下写出：

| 路径 | 内容 | 大小 |
|---|---|---|
| `checkpoints/step-XXXXXX-epoch-XXX-loss=X.XXXX.pt` | 模型权重 + **优化器状态** + scheduler + global_step + epoch | ≈ 14.75 GB |
| `checkpoints/step-XXXXXX-…safetensors` | 合并后的模型权重（推理用） | ≈ 14.47 GB |
| `checkpoints/step-XXXXXX-…-adapter.safetensors` | 与该 step 同步的 LoRA 适配器副本 | ≈ 145 MB |
| `checkpoints/latest-checkpoint.{pt,safetensors}` | 指向最新一份的软链（不参与滚动） | — |
| `adapter_model.safetensors` + `adapter_config.json` | 最新 LoRA 适配器（每次保存覆盖） | ≈ 145 MB |
| `tokenizer/`、`llm_backbone_config.json`、`README.md` | 配套文件（覆盖写，只有一份） | 小 |
| `<run_id>.jsonl` / `<run_id>.csv` | 训练指标（见下文「训练指标文件」） | 几 MB / 几十 KB |

**滚动清理**：`max_keep_ckpts=3`，按 mtime 删除最旧的 `.pt`，并同步删除同名 `.safetensors` 与 `-adapter.safetensors`。稳态占用 3 × 29.2 GB ≈ **87.6 GB**，保存瞬间峰值 ≈ **116.8 GB**。

### 为什么 LoRA 存档是「合并后」格式

`.pt` / `.safetensors` 里的 `model` 来自 `merge_and_unload()`：

```python
base_vla = build_vla_from_cfg(self.cfg.model)
base_vla.from_pretrained()
merged_vla = PeftModel.from_pretrained(base_vla, save_dir)
merged_vla = merged_vla.merge_and_unload()
model_state_dict = merged_vla.state_dict()
```

得到的是 **LoRA 已折进 base 的普通权重**，key 形如：

```
llm_backbone.layers.0.mlp.down_proj.weight
```

而运行时模型被 PEFT 包装，key 形如：

```
base_model.model.llm_backbone.layers.0.mlp.down_proj.base_layer.weight
base_model.model.llm_backbone.layers.0.mlp.down_proj.lora_A.default.weight
base_model.model.llm_backbone.layers.0.mlp.down_proj.lora_B.default.weight
```

**两者逐字不匹配。**

### ⚠️ 历史 bug：续训静默丢掉全部 LoRA 权重

修复前 `ddp_train_runner._load_model_state` 是：

```python
self.vla.module.load_state_dict(checkpoint_model_state, strict=False)
```

`strict=False` 遇到上面那种 key 全不匹配的情况，**不报错、只丢弃**。实测 812 个张量一个都没加载进去，模型退回「原始 `pi05_base_bf16` + 随机初始化 LoRA」。

2026-10-03 实测对比：

| | 第一步 loss |
|---|---|
| 原训练 step 999（中断前） | 0.0106 |
| **修复前**续训 step 1001 | **0.3888**（≈ 训练起点 0.43） |
| **修复后**续训 step 1001 | **0.0100** ✅ |

**最阴险的地方**：优化器状态恢复是正常的（日志 `Matched 830/838`），错配的 Adam 动量会把参数快速拉回，loss 会在十几步内从 0.39 回落到 0.05 上下——**看起来"训练正常"，实际上前面 1000 步的 LoRA 成果已经丢了。**

### 修复后的恢复顺序

`_load_model_state` 现在按优先级恢复：

1. **LoRA 模式优先走 adapter**：用 `peft.set_peft_model_state_dict` 加载与检查点**同 step** 的 `-adapter.safetensors`（找不到则回退到 run 根目录的 `adapter_model.safetensors`）。这样 base（`pi05_base_bf16`）+ LoRA + 优化器三者完全自洽。
   - `missing_keys` 只会剩下 `base_layer` / `original_module` / 未包装层（如 `patch_embedding`），它们由 `from_pretrained()` 提供，**属正常**。
2. **兜底**：adapter 不存在时，用 `_remap_merged_checkpoint_keys` 把合并 key 映射到 `base_model.model.<path>.base_layer.<param>`，而不是静默丢弃。

启动时会明确打印：

```
LoRA adapter restored from .../step-001000-epoch-000-loss=0.0409-adapter.safetensors
```

### 训练指标文件（jsonl / csv）

`metric` 配置：

```python
metric=dict(
    type='VLAMetric',
    active_trackers=('jsonl', 'csv'),
    csv_interval=100,
    run_dir='work_dirs',
    window_size=1,
)
```

| 文件 | 频率 | 说明 |
|---|---|---|
| `<run_id>.jsonl` | **每步**一条 | 全量指标，`run_id` = config 名 + 启动时间戳 |
| `<run_id>.csv` | **每 100 步**一行 | 便于肉眼/Excel 查看，列与 jsonl 相同，遇新列自动扩展 |

csv 列（当前配置 8 列）：

```
VLA Train/Step, Epoch, Loss, L1 Loss, Action Token Accuracy,
Loss (Raw), Learning Rate, Step Time
```

> `L1 Loss` 与 `Action Token Accuracy` 在当前 π0.5 流匹配实现下恒为 0（模型不产出这两个量）。关注 `Loss` / `Loss (Raw)` / `Learning Rate` / `Step Time` 即可。
> csv 在第 100 步之前**不会创建文件**，属正常。

### 续训命令

```bash
cd /home/lab/tron_ws/FluxVLA
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/lab/miniconda3/envs/fluxvla/bin/torchrun \
  --standalone --nnodes 1 --nproc-per-node 2 \
  scripts/train.py \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora.py \
  --work-dir work_dirs/tron2_buttons_v2 \
  --resume-from work_dirs/tron2_buttons_v2/checkpoints/step-001000-epoch-000-loss=0.0409.pt
```

注意事项：

- **必须用 `.pt`**（含优化器/调度器状态），不是 `.safetensors`。
- `pod_scripts/train_tron2_buttons.sh` **不支持 `--resume-from`**，续训要直接跑 `torchrun`。该脚本会就地 patch config，所以 config 已是可用状态，直接跑即可。
- 续训会生成**新的** `<时间戳>.jsonl` / `.csv`（`run_id` 含启动时间戳），**不会追加**到旧文件。看完整曲线需要把两段接起来。
- `run-metrics.jsonl` 会被新进程覆盖重写（不含曲线，无影响）。
- `global_step` 从检查点继续，通常无需改 `max_steps`。

### 判断续训是否正常

只看**第一步的 loss**：

```bash
tr '\r' '\n' < 训练日志 | grep "Global Step" | head -5
```

| 现象 | 结论 |
|---|---|
| 第一步 loss ≈ 中断前的量级（如 0.01） | ✅ 权重正确恢复 |
| 第一步 loss ≈ 训练起点（0.3 ~ 0.4） | ❌ 权重没恢复（就是上面那个 bug） |

---

## 已知坑与必读注意事项

### 1. 必须用 torchrun（否则 AttributeError）
见上一节。这是最容易踩的一个。

### 2. 数据加载必须用 `ParquetDatasetV3`，不能用 `ParquetDataset`

数采软件只能导出 **LeRobot v3.0**，而 FluxVLA 里存在**两个互不相通的数据族**：

| 族 | 数据集类 | 视频路径逻辑 | 适用格式 |
|---|---|---|---|
| v2.1 族 | `ParquetDataset` | `format(episode_chunk=, video_key=, episode_index=)` | v2.1 |
| **v3.0 族** | **`ParquetDatasetV3`** | 走 v3 模板 | **v3.0** |

用 `ParquetDataset` 配 v3.0 数据会在取样本时炸：

```
File "fluxvla/transforms/transform_inputs.py", line 174, in __call__
    video_root_path.format(episode_chunk=..., video_key=..., episode_index=...)
KeyError: 'chunk_index'
```

### 3. `info.json` 的 `video_path` 必须改用这三个占位符

```
原始 v3.0 模板（会 KeyError）：
  videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4

改为：
  videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4
  → 拼出 videos/observation.images.cam_high/chunk-000/file-000.mp4   ✅
```

**一行改动，不必重命名任何视频文件。**

### 4. 21 维必须在数据侧裁成 16 维
`NormalizeStatesAndActions` 的实现是 `_pad_or_truncate_last_dim`，**实际只 pad 不 truncate**。不裁剪的话头部/底盘/升降 5 维会被补零后混进动作。

### 5. 任务文本的零宽空格必须清洗
`\u200bPress the black button\u200b` 这类不可见字符会污染 prompt token 序列，使同义指令被当成不同序列。

### 6. **归一化 span 塌缩（本项目最大的坑）**
左臂静止导致 quantile 分母塌缩到 0.006 → 噪声放大 130~194 倍 → **98.8% 的目标方差被噪声占据**，18.17% 的窗口归一化后绝对值超过 10，实测最大 loss 尖峰 629。**必须使用修正后的 stats 文件**（见第 7 章）。

### 7. **回收站陷阱（会静默吃光磁盘）**
`rm` 被 shell 函数重定义进回收站；训练进程的 `os.remove` 被 `PYTHONPATH` 里的 `sitecustomize.py` 钩住。**必须 `unset PYTHONPATH` 启动 + 用 `/bin/rm` 清理**（见第 11 章）。

### 8. wandb 在无登录环境会挂起

`fluxvla/engines/metrics/vla_metric.py:106` 的 `wandb.init()` **没有 try/except**。此外 `WandBTracker.finalize()` 内含 `time.sleep(210)`。

**对策**：`active_trackers=('jsonl',)`。

### 9. `save_iter_interval` 必须能整除 `max_steps`

```python
def _should_save_step_checkpoint(self) -> bool:
    return (self.metric.global_step % self.save_iter_interval) == 0
```

**而且步骤态下这是唯一的保存路径**（epoch 保存只在 epoch 模式下生效），且训练循环结束时**没有收尾保存** —— 不整除则最后一步不落盘。

当前 `17000 % 1000 == 0` ✓

### 10. 腕部相机黑帧（数据侧问题，需重采解决）
约 30% 的 episode 中，腕部相机前 1–6 帧为纯黑。**下次采集务必先启动相机再开始录制。**

### 11. 依赖的代码补丁（勿随意 revert）

| 文件 | 改动 | 为什么必需 |
|---|---|---|
| `fluxvla/datasets/parquet_dataset_v3.py` | 新增 `supervise_terminal_padding` 参数 | 配置里传了该参数，原版没有 |
| `fluxvla/datasets/utils/transformed_statistics.py` | 自动统计支持 `ParquetDatasetV3` | 原版只认 `ParquetDataset`，会抛 ValueError |

`git checkout` 这两个文件会让训练跑不起来。

### 12. 部署侧的两个布局不一致（见第 14 章）
模型输出 16 维 `[L7,gripL,R7,gripR]`，而 `Tron2InferenceRunner` 期望 18 维 `[L7,R7,head2,gripL,gripR]`。

### 13. 环境侧（历史）
- `scripts/install_env.sh` 的 CUDA 版本判断过保守（只看 `nvidia-smi` 报告值），已打补丁
- flash-attn 必须源码编译（ABI 不匹配）
- 驱动 535.154.05 在宿主机内核，Pod 内无法修改

---

## 部署链路

### 三条部署路径

| 路径 | 入口 | 适用场景 |
|---|---|---|
| **ZMQ 远程推理**（推荐） | `scripts/zmq_inference_server.sh --config ... --ckpt-path ... --port 5555` | GPU 在服务器、机器人在现场。机器人侧只需 `pyzmq/msgpack/numpy`，**无需 GPU 与深度学习栈**；支持 SSH 隧道穿透 NAT；观测 JPEG 压缩传输 |
| **ROS 服务** | `scripts/ros_inference_server.sh --config ... --ckpt-path ... --service-name ... --node-name ... --ros-version 1\|2` | 给 FluxThemis 调用，支持 ROS1/ROS2 |
| **本机直跑** | 配置里的 `inference` 块（`Tron2InferenceRunner`） | GPU 与机器人同机 |

配套文档：

| 文档 | 内容 |
|---|---|
| `docs/remote_inference_serving.md` | ZMQ 架构、线格式（msgpack/protobuf）、SSH 隧道 |
| `docs/rtc.md` | Real-Time Chunking，改善 chunk 间轨迹连续性 |
| `docs/inference_acceleration.md` | Triton 融合核、CUDA Graph、自定义算子 |
| `docs/orin_docker_runtime.md` / `docs/orin_flashing.md` | Orin 板端部署 |

### 实测推理延迟（2026-09-30，本机 4090 D）

**延迟预算不是 33 ms，是 1067 ms** —— 一次推理输出 32 步动作，机器人按 30 Hz 执行：

```
控制频率 30 Hz  →  dt = 33.3 ms
动作块 32 步    →  一次推理要撑住 32 x 33.3 = 1067 ms 的机器人动作
```

**实测方法**：加载 `step-017000-*.safetensors`（`use_lora=False`，0 missing / 0 unexpected），
`torch.autocast('cuda', dtype=torch.bfloat16)`，warmup 3~5 次后计时 20 次取中位数。

| 路径 | 稳态延迟 | 占 1067 ms 预算 | 加速比 |
|---|---|---|---|
| **朴素** `PI05FlowMatching` | **236.3 ms**（范围 230.6 ~ 237.7） | 22.1% | 1x |
| **加速** `PI05FlowMatchingRTCInference` | **45.8 ms**（范围 45.7 ~ 45.9） | **4.3%** | **5.16x** |

**结论：延迟不是瓶颈 —— 连朴素路径都只占预算 22%。** 本机部署（零网络）下这一点更不成问题。

**加速比是 5.16x，不是文档宣称的 15x。** `docs/inference_acceleration.md` 的 15x 是在 **A100** 上标定的，**sm_89（4090）实测只有 5.16x**。

> **现成的部署配置**：`configs/pi05/pi05_paligemma_tron2_cabinet_lora_deploy.py`
> 用 `_base_` 复用训练配置，只覆盖 `inference_model`。已实测构建通过，可直接用于 `scripts/zmq_inference_server.sh`。

#### 三个实测踩到的坑

**① 非 RTC 版加速类在 3 路相机下会崩**

```
PI05FlowMatchingInference      ✗ RuntimeError: tensor a (2) vs b (3)   ← 3 路相机报错
PI05FlowMatchingRTCInference   ✓ 正常，45.8 ms
```

**必须用 `PI05FlowMatchingRTCInference`。** `docs/inference_acceleration.md` 的 π0.5 范例写的是非 RTC 版，**直接照抄会在我们的 3 相机配置上失败**。

**② 首次调用要 29.3 秒**（Triton JIT 编译 + CUDA Graph 捕获，日志会打 `[Triton Inference] Recording CUDA Graph ...`）。
**服务启动后必须先 warmup 一次再让机器人动作**，否则第一次推理会有 29 秒卡顿。

**③ 输入形状契约（与 docstring 不符，实测确认）**

```
images    : (B, num_views*3, H, W)     ← 注意是展平的，不是 (B, V, 3, H, W)
img_masks : (B, num_views)  bool
lang_tokens: (B, L)  int64
states    : (B, 32)
输出       : (B, 50, 32)  float32      ← n_action_steps=50，max_action_dim=32
```

### 三种部署形态对比

**结论：优先本机工作站单卡部署。**

| 形态 | 算力 | 图像开销 | 评价 |
|---|---|---|---|
| **本机工作站单卡** | 4090 | **0** | ✅ **首选** |
| 云服务器 + ZMQ | 4090 D | 211 KB/请求（可优化到 38 KB） | ✅ 可行，需 ≥5 Mbps 上行 |
| Orin 板端 | 弱很多 | 0 | 算力是瓶颈 |

> **两卡工作站的两卡带宽不一致不影响部署** —— 推理只需要 1 张卡，单卡串行前向，与第二张卡无关。
>
> ⚠️ **训练侧更正（2026-10-02 实测）**：这里原先写的"两卡并行比单卡还慢"**在 LoRA 配置下不成立**。
> GPU1 确实只有 PCIe 3.0 ×1（跨卡实测 0.82 GB/s），但 **LoRA 的梯度同步量极小，那条链路根本喂不满** ——
> 实测双卡 **1.84× 加速（92% 线性）**，每步只多 0.29 s（8%）。
>
> **"双卡更慢"只在全量微调口径下成立**：那时每步要同步 3B × fp32 = 12 GB 梯度，
> ring all-reduce ≈ 24 GB 流量 ÷ 0.82 GB/s ≈ **29 秒/步**，而一步算力本身只要 3.5 秒。
>
> 另需注意：**双卡不缩短单步，它让每步吃 2 倍数据** —— 要省总时长必须同时把 `max_steps` 减半
> （并同步改 `decay_steps`）。完整实测见 **`docs/tron2_dual_gpu_training_throughput.md`**。

#### 若用云服务器，图像开销可大幅压缩

模型只吃 **224×224**（`ResizeImagesWithPad`），但现在传的是 **480×640** —— 多传 6.1 倍像素，且 `JPEG_QUALITY = 95` 偏高。实测（真实数据帧，3 路合计）：

| 方案 | 载荷 | 2 Mbps 上行耗时 | 占 1067 ms 预算 |
|---|---|---|---|
| 现状 480×640 + q95 | 211.5 KB | 866 ms | 81% ⚠️ |
| 480×640 + q75 | 90.3 KB | 370 ms | 35% |
| **224×224 + q90** | **38.0 KB** | **155 ms** | **15% ✅** |

换成 224×224 + q90 后，2 Mbps 的普通宽带都能跑。

### 部署需要什么

```
<run_dir>/
├── checkpoints/step-XXXXX-epoch-XXX-loss=X.XXXX.safetensors    ← 合并后的完整模型 (14.5 GB)
└── dataset_statistics.json                                     ← 反归一化必需
```

推理代码**硬性要求** stats 文件在 `ckpt_path/../..`：

```120:123:fluxvla/engines/runners/base_inference_runner.py
data_stat_path = Path(Path(ckpt_path).resolve().parent.parent,
                      'dataset_statistics.json')
if not Path.exists(data_stat_path):
    raise ... f'Dataset statistics file not found at {data_stat_path}!'
```

**✓ 好消息**：训练启动时会把修正版 stats 导出到 `run_dir/dataset_statistics.json`，位置和内容自动正确，不需要手工搬运。

模型加载方式：

```python
if ckpt_path.endswith('.safetensors'):
    state_dict = load_file(ckpt_path, device='cpu')
...
self.vla.load_state_dict(state_dict, strict=True)
```

### Blocker：动作布局不一致

| | 布局 | 夹爪位置 | 维度 |
|---|---|---|---|
| **我们的训练数据 / 模型输出** | `[左臂7, 左夹爪, 右臂7, 右夹爪]` | **7, 15** | **16** |
| **`Tron2InferenceRunner` 期望** | `[左臂7, 右臂7, 头部2, 左夹爪, 右夹爪]` | **16, 17** | **18** |

代码依据（两处独立证据）：

```python
# tron2_inference_runner.py:94 —— prepare_pose 每条正好 18 个数
# [left(7), right(7), head_pitch, head_yaw, left_gripper(0-1), right_gripper(0-1)]
self.prepare_pose = [[1.2, 0, 0, -2.5, 0, 0, 0,  1.2, 0, 0, -2.5, 0, 0, 0,  0, 0,  1, 1], ...]

# tron2_inference_runner.py:284 —— 执行时按 18 维切片
left_arm_trajectory      = actions[:, :7]
right_arm_trajectory     = actions[:, 7:14]     # ← 我们的 7:14 实际是「左夹爪 + 右臂1~6」
head_trajectory          = actions[:, 14:16]
left_gripper_trajectory  = actions[:, 16]       # ← 16 维数组在此 IndexError
right_gripper_trajectory = actions[:, 17]
```

**直接用 16 维输出会崩，即使不崩也是命令错位。**

### 已实现的修法：`DenormalizeTron2Action`

**改动落在 `denormalize_action` 里，而不是 runner。** 因为本机直跑
(`BaseInferenceRunner._postprocess_actions`) 和 ZMQ 服务 (`zmq_server.py:258`)
**两条路径都会调用 `denormalize_action`** —— 放这里一处覆盖两条。

新增 `fluxvla/transforms/normalize.py::DenormalizeTron2Action`（继承 `DenormalizeDeltaAction`）：

```python
expanded[..., ROBOT_ARM_L]  = action[..., POLICY_ARM_L]   # 左臂   0:7  -> 0:7
expanded[..., ROBOT_ARM_R]  = action[..., POLICY_ARM_R]   # 右臂   8:15 -> 7:14
expanded[..., ROBOT_GRIP_L] = action[..., POLICY_GRIP_L]  # 左夹爪 7    -> 16
expanded[..., ROBOT_GRIP_R] = action[..., POLICY_GRIP_R]  # 右夹爪 15   -> 17
expanded[..., ROBOT_HEAD]   = current_head(data)          # 头部保持当前位姿
```

配置侧改成：

```python
denormalize_action=dict(
    type='DenormalizeTron2Action',     # ← 原来是 DenormalizeDeltaAction
    norm_type='quantile',
    action_dim=16,
    delta_action_mask=[True]*7 + [False] + [True]*7 + [False],
    state_permutation=[0,1,2,3,4,5,6, 16, 7,8,9,10,11,12,13, 17, 14,15],
)
```

### `state_permutation` 的长度必须是 18，不是 16

**这是实测推翻的一个关键点。** `DenormalizeDeltaAction` 的校验是：

```python
# normalize.py:476 —— 校验
expected = np.arange(self.state_permutation.size, dtype=np.int64)
if not np.array_equal(np.sort(self.state_permutation), expected):
    raise ValueError('state_permutation must contain every index in [0, D) exactly once.')
# normalize.py:496 —— 长度必须等于原始状态维度
if state.shape[-1] != self.state_permutation.size:
    raise ValueError('state_permutation length ... does not match raw state dimension ...')
state = state[self.state_permutation]
```

**它只能重排、不能筛选。** 机器人原始状态是 18 维，所以 permutation 必须是 18 的全排列。写成 16 直接抛错：

```
ValueError: state_permutation must contain every index in [0, D) exactly once.
```

正确写法把模型要用的 16 位放**前 16 位**（`delta_action_mask` 只消费 `state[:16]`），头部两位放末尾（不参与）：

```python
state_permutation = [0,1,2,3,4,5,6, 16, 7,8,9,10,11,12,13, 17, 14,15]
#                    └─ 左臂 ─┘ gripL └─── 右臂 ───┘ gripR  └头部┘
```

**不设或设错的后果（实测）**：dim8-14 会吃到机器人状态索引 8..14 = 右臂[1..6] + `head_pitch`，**右臂整体错位一格并混入头部值**。

### 左夹爪是个退化维度

实测统计量：

```
dim  7 (左夹爪): q01=0.0000  q99=0.0000  min=0  max=0  std=0    ← 全数据集恒为 0
dim 15 (右夹爪): q01=0.0000  q99=1.0000  mean=0.2142           ← 正常
```

**左夹爪在采集期间从未动过**，量化区间退化 → 反归一化把任何输入都塌成 0。所以左夹爪既学不到东西、输出也恒为 0。**部署时若需要左夹爪动作必须另行处理，不要指望模型。**

### 测试

`test/test_transforms/test_tron2_action_layout.py`（11 个用例）覆盖：16→18 映射、状态重排落位、
头部保持、夹爪不串位、**长度 16 的 permutation 必须报错**（防止回退到错误写法）、
已展开动作透传、以及从实际配置构建并验证映射。

### 部署前的风险清单

1. **stats 必须用修正版**（`tron2_stats_armsymmetric.json`）。若用回自动统计，左臂维度会按 150 倍尺度反归一化 → 左臂指令直接飙飞。
2. **左臂全程静止** → 模型对左臂没有有效学习信号。部署时建议对左臂输出做安全钳制，或直接锁定左臂。
3. **腕部相机黑帧** → 推理时若首帧为黑，预测会异常。建议部署侧加"首帧有效再推理"的检查。
4. **头部不参与**：`enable_head_control=False`，头部保持当前位姿。
5. **加速路径必须用 RTC 版类**：`PI05FlowMatchingRTCInference`。非 RTC 版 `PI05FlowMatchingInference` 在 3 路相机下会报 `tensor a (2) vs b (3)`（实测确认）。
6. **首次调用 29.3 秒**（Triton 编译 + CUDA Graph 捕获）。服务启动后**必须先 warmup 一次**再让机器人动作。
7. **输入形状与 docstring 不符**：`images=(B, nv*3, H, W)`（展平，不是 `(B,V,3,H,W)`）、`img_masks=(B, nv)`。

---

## 监控与产物

### 目录结构

```
work_dirs/tron2_cabinet_lora_v2/
├── config.json                                  # 完整生效配置（含所有默认值）
├── config.yaml                                  # 同上，YAML 格式
├── dataset_statistics.json                      # 训练实际使用的归一化统计量（= 修正版）
├── dataset_statistics_metadata.json             # 统计量元信息
├── run-metrics.jsonl                            # 运行级指标
├── train.log                                    # 训练日志
├── pi05_paligemma_tron2_cabinet_lora_<时间戳>.jsonl   # 逐步训练指标
└── checkpoints/
    ├── step-00X000-epoch-XXX-loss=X.XXXX.pt             # 含优化器/调度器，用于 resume
    ├── step-00X000-epoch-XXX-loss=X.XXXX.safetensors    # 合并后完整模型，用于部署
    ├── latest-checkpoint.pt -> ...
    └── latest-checkpoint.safetensors -> ...
```

> 注意：`checkpoints/` 必须是**真实本地目录，不能是软链**（曾临时软链到 s3fs，因慢 10 倍已撤回）。

### 逐步指标字段

```json
{
  "VLA Train/Step": 3000,
  "VLA Train/Epoch": 1,
  "VLA Train/Loss": 0.0402,
  "VLA Train/Loss (Raw)": 0.0402,
  "VLA Train/Learning Rate": "0.0000241",
  "VLA Train/Step Time": 4.23,
  "VLA Train/L1 Loss": 0.0,
  "VLA Train/Action Token Accuracy": 0.0
}
```

> `L1 Loss` 与 `Action Token Accuracy` 恒为 0，属该指标实现的既有情况，不代表训练异常。

### 实时查看进度

```bash
cd /root/tron_ws/FluxVLA
F=$(ls -t work_dirs/tron2_cabinet_lora_v2/pi05_paligemma_tron2_cabinet_lora_*.jsonl | head -1)
/opt/miniconda3/envs/fluxvla/bin/python -c "
import json, glob, statistics
f = sorted(glob.glob('work_dirs/tron2_cabinet_lora_v2/pi05_paligemma_tron2_cabinet_lora_*.jsonl'))[-1]
ls = [json.loads(l) for l in open(f) if l.strip()]
n = len(ls)
print('步数 %d / 17000 (%.2f%%)  剩余约 %.1f 小时' % (n, 100.0*n/17000, (17000-n)*4.25/3600))
print('loss 中位 %.4f  范围 %.4f ~ %.4f' % (
    statistics.median([x['VLA Train/Loss'] for x in ls]),
    min(x['VLA Train/Loss'] for x in ls), max(x['VLA Train/Loss'] for x in ls)))
for i in range(0, n, 500):
    w = [x['VLA Train/Loss'] for x in ls[i:i+500] if x['VLA Train/Loss'] <= 1]
    if w: print('  step %5d-%5d  均值 %.4f' % (i+1, i+len(ls[i:i+500]), statistics.mean(w)))
"
```

**看趋势要看滚动均值，不要看单步 loss** —— 单步 loss 天然有 ±40% 抖动（随机批次 + 流匹配时间步 `t` 每步重采样）。随收敛深入，抖动幅度会缩小但仍在。

### 关键观察点

| step | 观察什么 |
|---|---|
| 1000 | lr 达峰值 2.5e-5；**首次检查点保存（约 32 秒）** |
| 2354 | epoch 1 结束 |
| **4000** | **第 4 次保存触发滚动删除 → 验证磁盘空间真的释放** |
| 17000 | 训练结束，完整退火到 min_lr |

---

## 训练历史与对照实验

### 三次运行

| 运行 | 配置 | 步数 | 作用 |
|---|---|---|---|
| `work_dirs/tron2_v1` | batch=1 × accum=8（有效 8），**自动统计量** | 287 | 早期验证，loss 0.212 → 0.058 |
| `work_dirs/tron2_cabinet_lora_v1` | batch=3 × accum=8（有效 24），**自动统计量** | 3,068 | A/B 对照的基线（含尖峰问题） |
| **`work_dirs/tron2_cabinet_lora_v2`** | batch=3 × accum=8（有效 24），**修正统计量** | 进行中 | 正式训练 |
| `work_dirs/_v2a_300steps_reference` | 同上（未 unset PYTHONPATH 的那次） | 300 | 已归档 |

### A/B 对照：归一化修正的效果

两次运行仅在统计量上不同，其余完全一致：

| 区间 | v1 均值 | v2 均值 | 变化 |
|---|---|---|---|
| step 1–50 | 0.3891 | 0.3897 | +0.1% ← 起始一致，证明是干净单变量对照 |
| step 51–100 | 0.2872 | 0.2467 | **-14.1%** |
| step 101–150 | 0.2189 | 0.1451 | **-33.7%** |
| step 151–189 | 0.1805 | 0.1069 | **-40.8%** ← 差距持续拉大 |
| **整体** | **0.2725** | **0.2288** | **-16.0%** |

| 指标 | v1 | v2 |
|---|---|---|
| 尖峰（loss>1）次数 | 4 | **0** |
| 最大值 | **11.993** | **0.462** |

**初始 loss 相近是正常的** —— 两者都由"基座模型初始输出"决定。区别在于地板和尖峰：

| | 自动统计量 | 修正统计量 |
|---|---|---|
| 左臂目标值 | 0.5 ± 0.7（放大后的噪声） | 0.0045（≈0） |
| loss 地板 | 噪声不可降 | 模型学会输出 0 后地板为 0 |
| 尖峰 | 有（最高 629） | 无 |

### 历史 loss 参考（`tron2_v1`）

| step | loss | lr |
|---|---|---|
| 1 | 0.2116 | 0.0000000 |
| 21 | 0.2122 | 0.0000005 |
| 101 | 0.1252 | 0.0000025 |
| 201 | 0.1026 | 0.0000050 |
| 287 | **0.0578** | 0.0000072 |

该次运行在 warmup 未走完（lr 仅 7.2e-6 / 目标 2.5e-5）时 loss 已降到 0.058。

### 单步 loss 尖峰的既有现象

尖峰是这类流匹配训练的固有特性 —— `tron2_v1` 的 287 步里也有（>1 占 1.05%，最大 **15.5**）。原因是每步含多个样本，命中离群样本的概率随 batch 上升（实测 v1 在有效 batch 24 下 >2 的尖峰占 1.85%，正好是有效 batch 8 时的约 3 倍）。

**修正统计量后尖峰已消除** —— 说明之前 100% 的尖峰来自左臂归一化放大。

---

## 附：参考资源

| 资源 | 地址 |
|---|---|
| TRON2 预训练权重（7 个任务） | `https://hf-mirror.com/models/limxdynamics/tron2-openpi-models` |
| π0.5 基座权重 | `https://hf-mirror.com/models/limxdynamics/FluxVLAEngine` |
| 官方数据集 | `https://hf-mirror.com/datasets/limxdynamics/FluxVLAData` |
| 官方 v3.0 示例数据集 | `.../FluxVLAData/ARM_manual_test_10Episodes_lerobotv3.0` |

> `RealRobot_Tron2_lerobot/tron2_example`（config 里的默认路径）**未公开**，属厂商内部数据 —— 本配置已改用我们自己的 5 个数据集。
>
> 网络注意：`huggingface.co` 不可达，`hf-mirror.com` 可用，下载需走镜像。
>
> 可用的 TRON2 预训练权重：`pi05_tron2_banana` / `beans` / `candy` / `chassis` / `cloth` / `desktop` / `sort`，其中 `pi05_tron2_desktop` 与柜内桌面场景最接近，可考虑作为对照基线。

### 本机脚本清单

| 路径 | 作用 |
|---|---|
| `/root/tron_ws/scripts/check_lerobot_dataset.py` | LeRobot v2.1/v3.0 数据集体检（含零宽字符、夹爪、时间对齐、episode_success） |
| `/root/tron_ws/scripts/prepare_tron2_for_fluxvla.py` | 21 维 v3.0 数据 → 16 维 FluxVLA 可加载形态（含 video_path 改写、维度裁剪、文本清洗） |
| `/root/tron_ws/scripts/sweep_batch_ckpt.sh` | batch/梯度检查点扫描 |
| `/root/tron_ws/scripts/sweep_speed.sh` | 吞吐扫描 |
| `/root/tron_ws/scripts/sweep_dataloader.sh` | dataloader 扫描 |
