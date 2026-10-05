# TRON2 多任务数据集训练指导说明

> 编写日期：2026-10-04
> 适用工程：`/home/lab/tron_ws/FluxVLA`
> 适用配置：`configs/pi05/pi05_paligemma_tron2_cabinet_lora.py`
> 相关文档：`docs/pi05_tron2_cabinet_lora_training_config.md`、`docs/tron2_cabinet_lora_rebuild_guide.md`

本文所有结论都基于对本仓库代码的实读 + 对现场数据集/训练日志的实测，关键位置都标了 `文件:行号`。

---

## 0. TL;DR 快速决策表

| 你的情况 | 该怎么做 |
|---|---|
| 新增子任务与现有任务**同本体、同动作维度、同相机、同 fps** | **联合训练**（最省事、不遗忘）；只需处理统计 + 采样配比 |
| 新增子任务只差**任务语义**（都是按按钮类） | 联合训练，`task_descriptions` 按任务配 prompt |
| 新增子任务**时长尺度差很多** | 联合训练，horizon 取折中值（按第 6 节的掩码比例表选） |
| 新增子任务**动作量纲/范围差很多** | 联合训练 + **分组统计**（grouped datasets） |
| 新增子任务**动作维度 / 相机路数 / fps / 本体**不同 | ⚠️ 联合需要大量 padding，收益低 → 考虑顺序微调或分开训 |
| 数据量差异大，怕小任务被淹没 | 换 `DistributedBalancedRepeatingDataset` 做均衡采样（见 9.2~9.6） |
| 想区分各任务学得怎么样 | 训练只输出混合 loss，需自行加逐样本 loss + 按任务分组指标（见 9.7~9.9） |
| 只想改窗口大小 | dataset 的 `action_window_size` + 模型的 `n_action_steps` **成对改** |

---

## 1. 三层/五层 horizon 与约束链

### 1.1 完整清单

| # | 层 | 字段 | 默认 | 谁读 | 作用 |
|---|---|---|---|---|---|
| ① | **数据采样** | `action_window_size` | 9 | `ParquetDataset.__getitem__` | 每个训练样本监督**未来多少步动作**；决定越界填充比例 |
| ② | **目标对齐** | `action_horizon` | 40 | `PrepareStateActionTargets` | 把不同数据源的窗口 **pad 到同一长度**（补 0 + mask 补 0），保证 collator 能 stack |
| ③ | **模型产出** | `n_action_steps`(PI0/PI0.5)、`action_horizon`(DiT4DiT/GR00T head) | 50 / 40 | 推理时 `sample_actions` | **推理时固定生成多少步**（初始噪声形状）；训练时不读 |
| ④ | **执行** | `action_chunk` | 32 | `base_inference_runner._postprocess_actions` | 生成序列里**只下发前多少步**再重规划 |
| ⑤ | **执行(RTC)** | `execute_horizon` | — | `base_inference_runner.py:377-379` | 覆盖 ④，控制时间指针推进量 |

### 1.2 约束链

```
① action_window_size  ≤  ② action_horizon          （启用 ② 时，只能补不能截）
① action_window_size  == ③ n_action_steps          （建议对齐，否则 train/infer 分布不一致）
④ action_chunk        ≤  ③ n_action_steps
⑤ execute_horizon     ≤  ③ n_action_steps          （RTC）
```

硬报错位置：

- `fluxvla/transforms/transform_inputs.py:290-293`：`actions.shape[0] > action_horizon` → **raise**
- `fluxvla/collators/dict_collator.py:51-59`：`np.stack` → batch 内长度不等 → **raise**
- `fluxvla/datasets/utils/transformed_statistics.py:527-535`：多数据源 `action_window_size` 不一致 → **raise**（自动统计路径）

### 1.3 为什么必须分层

- **① 是数据语义**：定义「看到 `s_t` → 预测 `[a_t … a_{t+h-1}]`」的时间跨度。改它 = 改任务定义。
- **② 是工程适配**：只为了让多个数据源能塞进同一个 batch。单数据源且窗口一致时它是 **no-op**。
- **③ 是推理接口**：对外承诺一次给多少步。训练时模型从 runtime shape 推导，不需要固定值：

  ```130:136:fluxvla/models/vlas/pi05_flowmatching.py
      # Derive the mask from the supplied action window
      # because evaluation and truncated episodes can use a horizon
      # different from the training configuration.
  ```

  但推理必须固定形状：`pi0_flowmatching.py:844-846` → `actions_shape = (bsize, self.n_action_steps, self.max_action_dim)`

---

## 2. 当前训练的数据组织（实测）

### 2.1 现实结构

```python
# configs/pi05/pi05_paligemma_tron2_cabinet_lora.py:164-240
dataset=dict(
    type='DistributedRepeatingDataset',
    datasets=[                        # ← list，只有 1 个元素
        dict(
            type='ParquetDatasetV3',
            data_root_path=[          # ← 这一个实例里有 3 个根目录
                './datasets/RealRobot_Tron2_lerobot/lerobot_2026-10-02_22-55-58',
                './datasets/RealRobot_Tron2_lerobot/lerobot_2026-10-02_23-24-38',
                './datasets/RealRobot_Tron2_lerobot/lerobot_2026-10-02_23-53-41',
            ],
            action_window_size=50,
            ...
        )
    ])
```

**结论：训练用的是「一个数据集」，其中含 3 个子数据集 / 3 个任务。**

3 个根目录在 `ParquetDatasetV3.__init__` 里被拼成一个 HF dataset：

```196:204:fluxvla/datasets/parquet_dataset_v3.py
        for root in data_root:
            hf_dataset = load_dataset('parquet', data_dir=root, split='train')
            dataset_sizes.append(len(hf_dataset))
            datasets.append(hf_dataset)
        self.dataset_cumulative_sizes = np.cumsum([0] + dataset_sizes)
        self.dataset = concatenate_datasets(datasets)
        self.full_length = len(self.dataset)
```

### 2.2 什么共享 / 什么区分

| 项目 | 3 个子数据集之间 |
|---|---|
| `action_window_size` | 🔗 共享（一个值 50） |
| transforms 链 | 🔗 共享（同一套） |
| 归一化统计 | 🔗 共享（都读 `statistic_name='private'`） |
| episode 边界 / 串窗检查 | ✅ 独立（`episode_index` + `dataset_idx` 双重校验） |
| `task_description`（prompt） | ✅ 独立（各读自己的 `meta/tasks.parquet`） |

### 2.3 三条实测证据

1. `work_dirs/tron2_buttons_v2/dataset_statistics.json` 顶层 key 只有 `['private']`（`.private.proprio.mean` / `.private.action.mean` 各 16 维）→ 共用一份统计
2. 日志 `steps/epoch=1908, batch=96` → 1908×96 = 183,168 ≈ 三个根目录总帧数 **183,114**
3. `__len__` 取的是 `self.full_length`（帧总数，不是帧数−集数）：

   ```204:206:fluxvla/datasets/parquet_dataset_v3.py
           self.full_length = len(self.dataset)
           self.sample_indices = np.arange(self.full_length, dtype=np.int64)
           self.effective_length = self.full_length
   ```

   每集的**最后一帧**作为起点会在 `__getitem__` 里被重采样（`261-277` 行），所以样本总数 = 总帧数。

---

## 3. 四种数据组织方式

`DistributedRepeatingDataset` 支持三种格式，加上 balanced 变体共四种：

```39:45:fluxvla/datasets/dataset_wrapper.py
    Now supports three formats:
    1. Single dataset (dict): Single dataset configuration
    2. List of datasets (list of dict): Treats all datasets as one
        concatenated dataset
    3. Grouped datasets (dict of list of dict): Groups datasets
        by keys, each group has separate statistics and is
        treated as a separate dataset
```

| 写法 | 独立窗口 | 独立 transforms | 独立统计 | 均衡采样 | 只建一套 transform |
|---|---|---|---|---|---|
| **A. 单 dict + 多 root**（现状） | ❌ | ❌ | ❌ | ❌ | ✅ |
| **B. `datasets=[d1,d2,d3]`** | ✅ | ✅ | ❌（共用一份） | ❌ | ❌（每个建一套） |
| **C. grouped `datasets={'g1':[d]…}`** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **D. balanced + 单 dict 多 root** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **E. balanced + `datasets=[d1,d2,d3]`** | ✅ | ✅ | ❌ | ✅ | ❌ |

### 窗口是实例级参数：想按任务区分窗口只能用多实例

`action_window_size` 是**实例级**参数，一个 `ParquetDataset` 实例内所有 `task_index` 共享它，**框架没有按任务区分窗口的入口**。

| 你的组织方式 | 各任务窗口能不同吗 |
|---|---|
| A / D（单实例多 root） | ❌ 共享，无法区分 |
| B / C / E（多实例） | ✅ 每个实例配自己的 `action_window_size` |
| 任何方式，但窗口统一（如都 50） | — 无需区分 |

**结论**：确实需要每个任务用不同窗口时，只能用**多实例**（B/C/E）组织数据，不要指望在单实例多 root 里做任务级区分。

> ⚠️ 用 B/C/E 且各任务窗口**不同**时，**必须启用 `PrepareStateActionTargets(action_horizon = max(各窗口))`**（配置层改动），否则 batch 内长度不等，`np.stack` 直接崩。注意 transforms 是**每个 dataset 一份**的，所以要写进每个 dataset 的 `transforms` 列表（配置里可以用同一个 Python 变量复用）。

### E 是「每任务独立 + 均衡采样」的最优组合

balanced wrapper 的 `datasets` 接受 list，且 `is_list=True` 时**每个 list item 就是一个 source**：

```114:119:fluxvla/datasets/balanced_dataset_wrapper.py
        if self.is_list:
            return [
                np.arange(length, dtype=np.int64)
                for length in self.dataset_lens
            ]
```

所以 `datasets=[d1,d2,d3]` + `DistributedBalancedRepeatingDataset` 能同时拿到：
- 每个子任务自己的 `action_window_size` / transforms / `statistic_name`
- 每任务等权的均衡采样（或 `sampling_weights` 加权）

代价：每个数据集建一套 transform/tokenizer pipeline（内存 + 启动时间）。

反过来，框架提供「多 root 单数据集」作为 source 的设计意图正是**为了省掉这套重复建设**：

```37:39:fluxvla/datasets/balanced_dataset_wrapper.py
    A source can be either an item in a dataset list or one root of a single
    multi-root :class:`ParquetDataset`. Supporting the latter avoids building
    a tokenizer and transform pipeline once per RoboCasa task.
```

**选型口诀**：任务同质（同一套 transforms、同一份统计、同一个合适窗口）→ 用 A/D 省资源；任务异质（不同窗口/不同统计）→ 用 B/C/E 换灵活性。

**B/C 的统计行为**

- list 模式：`dataset_statistics_path` 会**覆盖全部数据集，且只有一份统计**（`dataset_wrapper.py:157-161`）→ 想每任务不同统计要用 `statistics_overrides`
- grouped 模式：**禁止 `dataset_statistics_path`**：

  ```164:168:fluxvla/datasets/dataset_wrapper.py
          if dataset_statistics is not None:
              raise ValueError(
                  'dataset_statistics_path is only supported for '
                  'single/list dataset configs; grouped datasets should use '
                  'grouped stats.')
  ```

  grouped 的每组统计由 `get_dataset_statistics` 各自算（`206-209`）

**D 的坑：必须去掉外层 list**

`DistributedBalancedRepeatingDataset._build_source_positions` 只有在**非 list** 模式下，才会把多根数据集的每个 root 当成一个 source：

```114:138:fluxvla/datasets/balanced_dataset_wrapper.py
        if self.is_list:
            return [
                np.arange(length, dtype=np.int64)
                for length in self.dataset_lens
            ]

        cumulative_sizes = getattr(self.dataset, 'dataset_cumulative_sizes',
                                   None)
        ...
        return [
            positions[(sample_indices >= start) & (sample_indices < end)]
            for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:])
        ]
```

- `datasets=[dict(多根)]` → `is_list=True` → **只有 1 个 source**（等于没均衡）
- `datasets=dict(多根)` → `is_list=False` → **每个 root 一个 source** ✅

另外 balanced 变体**不支持 grouped**：

```86:89:fluxvla/datasets/balanced_dataset_wrapper.py
        if self.is_grouped:
            raise ValueError(
                'DistributedBalancedRepeatingDataset does not support '
                'grouped datasets.')
```

**D 的默认行为就是"每个任务等权"**：不给 `sampling_weights` 时，每个 source 每轮各贡献一次。

---

## 4. 添加新子任务：8 步流程

### Step 1 — 差异体检（先做这个，决定后面所有事）

逐项对比新旧数据集：

| 检查项 | 怎么看 | 不一致的后果 |
|---|---|---|
| 动作维度 | `action` 向量长度 | ❌ 最硬：collator 直接崩；需 pad 到同一 `max_action_dim` |
| 状态维度 | `observation.state` 长度 | 同上（`state_dim`） |
| 相机路数/命名 | `meta/info.json` 的 `features` | ⚠️ 需 image padding + `img_masks` |
| fps | `info.json` 的 `fps` | ⚠️ 必须重采样到统一频率，否则动作速度语义混乱 |
| 机器人本体 | 人工确认 | ❌ 完全不同本体不建议联合 |
| 动作量纲/范围 | 对比 q01/q99 | ⚠️ 需分组统计，否则互相污染 |
| episode 长度分布 | `meta/episodes/*.parquet` 的 `length` | ⚠️ 影响 padding 比例与 horizon 选择 |
| 数据质量 | `episode_success` / `complementary_info.is_intervention` | 低质量数据会拖累联合训练 |

**体检脚本** 见第 12 节。

### Step 2 — 选组织方式

按第 0 节决策表 / 第 3 节能力矩阵选 A/B/C/D/E。

### Step 3 — 处理 horizon

- 各任务窗口**一致** → 什么都不用做
- 各任务窗口**不同** → 必须启用 ② `PrepareStateActionTargets`，且 `action_horizon = max(各窗口)`
- 想**每个任务用自己的最优窗口** → 换成多实例组织（B/C/E），每个实例各配 `action_window_size`（见第 3 节）

### Step 4 — 处理统计（最容易出错，见第 8 节）

### Step 5 — 处理采样配比（见第 9 节）

### Step 6 — 对齐推理 prompt（见第 10 节）

### Step 7 — 重算 `steps_per_epoch` 与 `max_steps`

```
样本数 ≈ 各数据集总帧数之和
steps_per_epoch = 样本数 / (per_device_batch_size × world_size × grad_accumulation_steps)

# 本机实测
per_device_batch_size=3, world_size=2, grad_accumulation_steps=16
→ global batch = 96
→ 当前 183,114 帧 → steps/epoch = 1908
```

> ⚠️ 注意：该配置文件的注释里写的是「effective batch 24」，与实际不符。以日志打印的 `Global (Effective) Batch` / `steps/epoch` 为准。

### Step 8 — 从头训 + 验证

改统计 / 改窗口 / 加数据集 → **旧 checkpoint 全部不兼容**，必须从头开新 `--work-dir`。验证方法见第 12 节。

---

## 5. 现场数据集实测档案（2026-10-04）

### 5.1 当前训练用（`datasets/RealRobot_Tron2_lerobot/`）

| 数据集 | 任务 | episodes | 帧数 | ep 长度 min/中位/max | success | 干预 |
|---|---|---:|---:|---|---|---:|
| `..._10-02_22-55-58` | Press the **red** button | 506 | 61,828 | 51 / 123 / 246 | 100% | 0% |
| `..._10-02_23-24-38` | Press the **black** button | 503 | 64,041 | 54 / 124 / 262 | 100% | 0% |
| `..._10-02_23-53-41` | Press the **green** button | 504 | 57,245 | 51 / 105 / 242 | 100% | 0% |
| **合计** | | **1513** | **183,114** | | | |

### 5.2 `_archive/` 里被归档的 5 个

| 数据集 | 任务 | episodes | 帧数 | ep 长度 min/中位/max | success | 干预 |
|---|---|---:|---:|---|---|---:|
| `..._09-28_22-05-37` | Press the **red** button | 102 | 8,834 | 51 / 83 / 140 | 100% | 0% |
| `..._09-28_22-26-43` | Press the **black** button | 104 | 8,242 | 54 / 79 / 108 | 100% | 0% |
| `..._09-28_22-36-10` | Press the **green** button | 107 | 8,097 | 51 / 75 / 103 | 100% | 0% |
| `..._09-28_22-44-32` | **Switch the selector L→R** | 134 | 17,875 | 84 / 130 / 249 | 100% | 0% |
| `..._09-28_22-54-05` | **Switch the selector R→L** | 109 | 13,437 | 97 / 122 / 155 | 100% | 0% |
| **合计** | | **556** | **56,485** | | | |

### 5.3 兼容性体检结论（Press ↔ Switch）

| 检查项 | Press | Switch | 兼容 |
|---|---|---|---|
| action / state 维度 | 16 / 16 | 16 / 16 | ✅ |
| 相机 | `cam_high` + `cam_left_wrist` + `cam_right_wrist` | 完全同名同路数 | ✅ |
| fps | 30 | 30 | ✅ |
| features key | 一致 | 完全一致 | ✅ |
| 本体 | TRON2 | TRON2 | ✅ |
| ep 长度中位 | 118 | 125 | ✅ 接近 |
| 任务语义 | 按按钮 | 拨开关 | ❌ 不同 → 靠 prompt 区分 |

**结论：只有任务语义（和轻微时长尺度）不同 → 联合训练是正确选择。**

### 5.4 合并后的规模

| 组合 | episodes | 帧数 | steps/epoch（batch=96） |
|---|---:|---:|---:|
| 现状（3× Press） | 1513 | 183,114 | 1908 |
| + 2× Switch | 1756 | 214,426 | ≈ 2234 |
| + 2× Switch + 3× 旧 Press | 2069 | 239,599 | ≈ 2496 |

> 旧 Press（09-28）episode 更短（中位 75~83），混入会引入"演示节奏"差异，建议先只加 Switch，验证 OK 后再加。

---

## 6. horizon 怎么选

### 6.1 掩码比例公式

集长 `L`、窗口 `h`（`h ≤ L`）：

```
起点范围      s ∈ [0, L-2]              （末帧会被重采样排除）
可用帧数      R = L - s ∈ [2, L]
溢出步数      o = max(0, h - R)
单集溢出总和  Σ o = (h-1)(h-2)/2
掩码步占比    = (h-1)(h-2) / (2 · h · (L-1))   ≈  h / (2L)
```

### 6.2 实测表（按真实 episode 长度分布计算）

**现状 3× Press（1513 eps / 183,114 帧）**

| `h` | 前瞻 | 掩码步占比 | 带掩码样本占比 | 平均真实步数 |
|---:|---:|---:|---:|---:|
| 16 | 0.53 s | 5.5% | 11.7% | 15.1 |
| 20 | 0.67 s | 7.1% | 15.0% | 18.6 |
| 24 | 0.80 s | 8.8% | 18.3% | 21.9 |
| **30** | **1.00 s** | **11.3%** | 23.3% | 26.6 |
| **40** | **1.33 s** | **15.4%** | 31.7% | 33.8 |
| **50（现用）** | **1.67 s** | **19.6%** | 40.0% | 40.2 |
| 64 | 2.13 s | 25.4% | 51.6% | 47.7 |
| 80 | 2.67 s | 32.0% | 64.2% | 54.4 |

**3× Press + 2× Switch（1756 eps / 214,426 帧）**

| `h` | 前瞻 | 掩码步占比 |
|---:|---:|---:|
| 24 | 0.80 s | 8.7% |
| 30 | 1.00 s | 11.2% |
| 40 | 1.33 s | 15.3% |
| 50 | 1.67 s | 19.4% |
| 64 | 2.13 s | 25.2% |

### 6.3 建议

- pi0 类模型常用 **1~2 秒**前瞻
- 只看 Press（整集约 4s）：`h=30` 或维持 50 都可接受
- 要同时照顾"快按"和"慢拨"（Switch 中位 125 帧 ≈ 4.2s）：**`h` 取 30~40 更稳妥**
- `h` 越大 → 有效监督比例越低、单条样本成本越高；`h` 越小 → 重规划越频繁

---

## 7. 窗口内越界部分的真实行为（重要，容易搞错）

### 7.1 越界时**不是补 0**，是**重复最后一帧**，且 `mask=0`

两条分支，只有一条会被监督：

```308:323:fluxvla/datasets/parquet_dataset_v3.py
            elif not future_in_range or future_task == 'empty':
                for _ in range(self.action_window_size - len(actions)):
                    actions.append(actions[-1])
                    action_masks.append(
                        1 if self.supervise_terminal_padding else 0)
                break
            elif future_task == 'static':
                window_idx += 1
                continue
            else:
                if len(actions) > 0:
                    actions.append(actions[-1])
                else:
                    actions.append(data[self.action_key])
                action_masks.append(0)
            window_idx += 1
```

| 情况 | 动作值 | mask |
|---|---|---|
| 同集内正常帧 | 真实动作 | **1** |
| **跨到下一集**（越界的主因） | 重复最后真实动作 | **0**（硬编码，与 `supervise_terminal_padding` 无关） |
| 数据集末尾 / `task == 'empty'` | 重复最后真实动作 | `supervise_terminal_padding` |
| `task == 'static'` | 跳过不占位 | — |

**推论**：当 `task` 全是正常名字（无 `empty`）时，`supervise_terminal_padding=True` **几乎不起作用**，因为起点重采样（`261-277`）已经排除了会触发它的位置。

### 7.2 起点重采样规则

```262:271:fluxvla/datasets/parquet_dataset_v3.py
            if index == len(self.dataset) - 1:
                needs_resample = True
            else:
                next_data = self.dataset[index + 1]
                next_task = self._resolve_task_description(
                    dataset_idx, next_data)
                needs_resample = (
                    data['episode_index'] != next_data['episode_index']
                    or self._get_dataset_index(index + 1) != dataset_idx
                    or next_task in ('empty', 'static'))
```

只检查**紧邻下一帧**，所以：合法起点 = `[集首, 集尾-1]`，每个窗口**至少 2 步真实动作**。

### 7.3 各层 padding 的性质对照

| 位置 | 轴 | 填充值 | mask | 进 loss |
|---|---|---|---|---|
| dataset 越界 | 时间 | **重复最后一帧**（不是 0） | 0（跨集）/ `supervise_terminal_padding`（末尾、empty） | 多数不进 |
| `NormalizeStatesAndActions(action_dim=32)` | **动作维度 16→32** | **0** | 靠 `loss_action_dim=16` 排除 | ❌ |
| `PrepareStateActionTargets` | 时间（补到 `action_horizon`） | **0** | 显式补 0 | ❌ |

补 0 的两处都靠 mask/切片排除出 loss：

```271:272:fluxvla/models/vlas/pi0_flowmatching.py
            return (prediction[..., :self.loss_action_dim],
                    target[..., :self.loss_action_dim])
```

```38:63:fluxvla/engines/losses/rabc.py
    def reduce_action_bc_loss(losses: torch.Tensor,
                              action_mask: Optional[torch.Tensor] = None,
                              ...
        action_mask (Optional[torch.Tensor]): Valid-action mask. It can be
            shaped like ``[batch, horizon]`` or exactly like ``losses``.
    ...
        valid = _expand_to_losses(_to_loss_tensor(action_mask, losses), losses)
```

⚠️ **但补 0 / 重复帧那段仍然会作为输入进 attention**（loss 被 mask，输入没被 mask），这是唯一的副作用来源。

### 7.4 相对空间里的值

`RelativeActions` 是 **state-relative**（每个动作减**当前帧 state**），不是逐步差分：

```181:186:fluxvla/transforms/transform_actions.py
        states = np.asarray(data[self.state_key])
        actions = np.asarray(data[self.action_key]).copy()
        dims = self.mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(self.mask, states[..., :dims], 0), axis=-2)
        data[self.action_key] = actions
```

所以尾部那串"重复的绝对动作"归一化前是常量 `a_last − s_t`（14 个臂维）+ 常量 `a_last`（2 个夹爪维），**不是 0**。因为 mask=0，值是什么不影响梯度。

---

## 8. 统计：多任务训练最容易踩的坑

### 8.1 三条铁律

1. **同一份统计内，所有任务共享同一套 q01/q99**。如果某个任务里某维度几乎不动，`(x-q01)/(q99-q01)` 会把噪声放大。
   > 本项目已踩过一次：左臂在 5 个数据集里全程静止，相对动作 q01-q99 跨度塌缩到 ~0.006，放大约 150 倍，左臂吃掉 67.5% 的 loss、并造成最高 629 的 loss 尖峰。修正方案：左臂 0-6 维改用对称右臂 8-14 维的 q01/q99（`datasets/RealRobot_Tron2_lerobot/tron2_stats_armsymmetric.json`）。

2. **改动窗口 / 加数据集 / 加任务 → 统计必须重算**，且**旧 checkpoint 不兼容**（归一化空间变了）。

3. **顺序训练的两段必须共用同一份统计**（否则第二段把第一段的输出空间错位）。

### 8.2 处理方式

| 方式 | 做法 | 适用 |
|---|---|---|
| **共用一份（现状）** | `dataset_statistics_path='.../tron2_stats_X.json'` | 任务同质、量纲接近 |
| **人工修正** | 手工改某些维度的 q01/q99（如 armsymmetric） | 某维度在部分任务里静止 |
| **分组统计** | grouped datasets（每组独立算）或 list + `statistics_overrides` | 各任务量纲差异明显 |

### 8.3 必须检查的一项

对每个新任务单独算一遍「每维度的 q99-q01 跨度 / 帧间步进」，找出**接近常量**的维度。若新任务的"静止维度"与旧任务不同，共用统计一定会出问题。

---

## 9. 任务配比与损失结构

多任务训练里「每个任务影响训练的程度」由两件事决定：**采样时的任务占比**（9.1–9.6）和 **loss 的加权方式**（9.7–9.9）。两者作用位置不同，很容易混为一谈。

### 9.1 默认采样：按数据量加权

默认 `DistributedRepeatingDataset` 的索引就是一段连续全局序号再 shuffle，所以每个任务被采到的比例**天然等于它的样本占比**：

```600:614:fluxvla/datasets/dataset_wrapper.py
    def _get_round_robin_indices(self, epoch: int) -> np.ndarray:
        ...
        epoch_offset = epoch if self.reshuffle_each_epoch else 0
        indices = np.arange(self.total_len)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch_offset)
            rng.shuffle(indices)
        return indices
```

⚠️ 也就是说 **A/B/C 三种组织方式都不是均衡采样**，只有 `DistributedBalancedRepeatingDataset`（D/E）才是。

本项目 Press + Switch 的默认占比：

```
Press  183,114 帧  → 85.4%
Switch  31,312 帧  → 14.6%
```

### 9.2 「等权」是什么

一句话：**让每个任务分到相同的训练预算，而不是按数据量分。**

| 采样方式 | 每个任务占比 | 谁在用 |
|---|---|---|
| 按数据量（默认） | = 该任务帧数占比 | A / B / C |
| 按 source 等权 | 每个 root 各 20% | D / E 默认 |
| 自定义加权 | `sampling_weights` 连续可调 | D / E |

两种粒度：

- **按 source 等权**（D/E 默认）：5 个 root 各 20%
- **按任务组等权**：用 `sampling_weights` 自己定（例如 3 个 Press : 2 个 Switch = 50:50）

### 9.3 等权有什么用

1. **核心认知：训练预算 ≠ 数据量。** 一个任务需要多少次梯度更新，取决于它的**难度和多样性**，而不是"采了多少条"。Switch 只有 243 条 episode，不代表它"只要 14.6% 的训练量就能学会"——拨开关和按按钮是两个不同的动作类型。
2. **防止小任务被主流任务淹没。** batch=96 时按比例采样，平均每批只有约 **14 条**来自 Switch、**82 条**来自 Press → 每个 optimizer step 的梯度方向由 Press 主导。
3. **均衡任务间的最终性能。**

但必须清楚：**等权不创造额外训练量，它只是重新分配**。总预算够（每个任务都收敛）时用不用差别不大；只有总预算有限、某个任务确实欠训练时，等权才真的救它。

**判断方法**：先按比例训一轮，**分别评估各任务**（注意训练日志里看不到 per-task 指标，见 9.7）。哪个任务差就给哪个加权。

### 9.4 等权的代价

| 代价 | 说明 |
|---|---|
| 小数据集被重复采样 | 等权下 Switch R→L 每轮被看 **4.77 遍**，过拟合该任务的风险上升 |
| 大任务被稀释 | 同样 `max_steps` 下 Press 的 pass 数变少，Press 性能可能略降 |
| `epoch_size` 变大 | 要重算 `steps_per_epoch` 和 `max_steps`（见 9.10） |

### 9.5 三种配比对现场 5 个数据集的实测数字

| 配比方案 | red | black | green | SW-L2R | SW-R2L | Press:Switch | 每 source 每轮重复次数 | `steps/epoch`(batch=96) |
|---|---:|---:|---:|---:|---:|:--:|---|---:|
| ① 按帧（默认） | 28.8% | 29.9% | 26.7% | 8.3% | 6.3% | **85.4 : 14.6** | 全部 1.0× | 2,234 |
| ② 按 source 等权（D/E 默认） | 20% | 20% | 20% | 20% | 20% | **60 : 40** | 1.04 / 1.00 / 1.12 / **3.58 / 4.77** | 3,335 |
| ③ 按任务组等权（50:50） | 16.7% | 16.7% | 16.7% | 25% | 25% | **50 : 50** | 1.04 / 1.00 / 1.12 / **5.38 / 7.16** | 4,003 |

> ②③ 的"等权"本质是**按任务组平移预算**，Switch 每轮被重复 3.6~7.2 遍。

### 9.6 推荐做法：用 `sampling_weights` 连续调，不必二选一

对本项目（Switch 是新任务类型但只有 243 条），极端等权（40~50%）偏激进，建议折中：**Press : Switch = 75 : 25**

```python
dataset=dict(
    type='DistributedBalancedRepeatingDataset',
    datasets=dict(                    # 单 dict（去掉外层 list），每个 root 一个 source
        type='ParquetDatasetV3',
        data_root_path=[red, black, green, switch_l2r, switch_r2l],
        ...),
    # 顺序必须与 data_root_path 一致
    sampling_weights=[0.25, 0.25, 0.25, 0.125, 0.125],   # → Press 75% / Switch 25%
    shuffle=False, reshuffle_each_epoch=True, seed=42,
)
```

| | red | black | green | SW-L2R | SW-R2L |
|---|---:|---:|---:|---:|---:|
| 份额 | 25% | 25% | 25% | 12.5% | 12.5% |
| 每轮重复 | 1.04× | 1.00× | 1.12× | **1.79×** | **2.38×** |
| `epoch_size` | `max(len_i / p_i) = 256,164` → `steps/epoch ≈ 2668` | | | | |

**推荐流程**：

1. 先用 **75:25** 训一轮（Switch 重复 1.8~2.4×，过拟合风险可控）
2. 分别在 5 个任务上评估（尤其两个 Switch 任务）
3. Switch 明显差 → 提到 70:30；Press 明显退步 → 退回 80:20 或直接按帧

`sampling_weights` 的约束（必须一个 source 一个值、且为正数）：

```140:153:fluxvla/datasets/balanced_dataset_wrapper.py
    def _normalize_sampling_weights(
            self, sampling_weights: Optional[Sequence[float]]):
        ...
        if weights.shape != (len(self.source_lengths), ):
            raise ValueError(
                '`sampling_weights` must contain one value per source, got '
                f'{weights.shape} for {len(self.source_lengths)} sources.')
```

### 9.7 各任务的 loss 是单独计算的吗 —— **不是**

整个 batch（混合任务）只算**一个标量**：

```756:766:fluxvla/models/vlas/pi0_flowmatching.py
        v_t, u_t = self._select_flow_loss_dimensions(v_t, u_t)
        losses = F.mse_loss(u_t, v_t, reduction='none')
        sample_weight = kwarg.get('sample_weight')
        loss = reduce_action_bc_loss(
            losses, action_mask=action_masks, sample_weight=sample_weight)

        return_dict = dict(
            predictions=v_t,
            loss=loss,
        )
        return return_dict
```

- `losses` 形态是 `[B, horizon, loss_action_dim]`，B 里混着不同任务的样本
- `reduce_action_bc_loss` 按 mask 加权后**压成一个 scalar**
- 返回的 dict 里**只有 `loss`**，没有任何任务维度的拆分

日志里同样只有全局指标：

```
VLA Train/Step, Epoch, Loss, L1 Loss, Action Token Accuracy,
Loss (Raw), Learning Rate, Step Time
```

（顺带：`L1 Loss` 与 `Action Token Accuracy` **恒为 0**，实际没在算。）

### 9.8 稀释效应：为什么"总 loss 正常"不等于"每个任务都学会了"

`reduce_action_bc_loss` 的分母是有效元素总数，所以各任务的权重 ≈ 它的**有效元素占比**。对本项目：

| 任务组 | 样本占比 |
|---|---:|
| Press | 85.4% |
| Switch | 14.6% |

假设 Press 的 loss 是 0.006（当前量级），而 Switch 学得很差、loss 是 0.048（**差 8 倍**）：

```
混合 loss = 0.854 × 0.006 + 0.146 × 0.048 ≈ 0.0121
```

**Switch 差 8 倍，总 loss 只从 0.006 涨到 0.012（约 2 倍）** —— 训练日志上看不出异常，训完做推理才发现 Switch 不会用。

这也是"训练时唯一的反馈就是这一个混合 loss"的直接后果，它天然偏向数据量大的任务。

### 9.9 想看 per-task loss 怎么办

**路线 1：只加指标（推荐先做，不动训练语义）**

1. 模型 forward 里额外返回**逐样本** loss（不修改 `loss` 本身）：

   ```python
   # reduce 之前
   valid = action_masks.expand_as(losses)
   per_sample = (losses * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)
   return_dict = dict(predictions=v_t, loss=loss, per_sample_loss=per_sample)
   ```

2. runner / metric 里按任务分桶累加。**任务标签现成就有** —— collator 的 `meta_keys` 已经带上 `task_description`：

   ```280:280:configs/pi05/pi05_paligemma_tron2_cabinet_lora.py
           meta_keys=['task_description', 'prompt', 'info', 'stats']),
   ```

   按 `task_description` 字符串分桶即可，**数据集侧不用改**。

产出形如 `VLA Train/Loss/press_red`、`VLA Train/Loss/switch_l2r`。

**路线 2：按任务给 loss 加权**

框架已预留 per-sample 权重通道，从 transform 一路传到 loss：

```758:760:fluxvla/models/vlas/pi0_flowmatching.py
        sample_weight = kwarg.get('sample_weight')
        loss = reduce_action_bc_loss(
            losses, action_mask=action_masks, sample_weight=sample_weight)
```

```208:210:fluxvla/transforms/transform_inputs.py
        if 'sample_weight' in data:
            inputs['sample_weight'] = np.asarray(
                data['sample_weight'], dtype=np.float32)
```

现成的 `AttachRABCWeight` 是给 RA-BC 用的（按样本 index 查表打权），它的 docstring 说明了这条通道怎么走：

```36:43:fluxvla/transforms/attach_rabc_weight.py
class AttachRABCWeight:
    """Attach one RA-BC sample weight to each training sample.

    Put this transform before transforms that rebuild the sample dictionary,
    such as ``ProcessParquetInputs``. Those transforms can then carry
    ``sample_weight`` through to the collator.
    """
```

要按任务打权，写个几行的小 transform 即可：读 `task_index` → 查表 → 写 `sample_weight`。**效果等价于改采样比例，而且更精细**（每个 batch 内立即生效，不受"某个 batch 恰好没抽到 Switch"的影响）。

⚠️ 前提：`task_index` 得能传到 batch，而它现在被 `ProcessParquetInputs.parquet_keys` 过滤掉了：

```194:197:configs/pi05/pi05_paligemma_tron2_cabinet_lora.py
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
```

```141:141:fluxvla/transforms/transform_inputs.py
        for key in self.parquet_keys:
```

→ 需要在 `parquet_keys` 里加 `'task_index'`；若还要它进 tensor，得同时加到 collator 的 `keys`。

### 9.10 副作用：`epoch_size` 变了

```103:112:fluxvla/datasets/balanced_dataset_wrapper.py
        if epoch_size is None:
            if self.sampling_probabilities is None:
                epoch_size = len(self.source_lengths) * max(
                    self.source_lengths)
            else:
                lengths = np.asarray(self.source_lengths, dtype=np.float64)
                epoch_size = int(np.max(lengths / self.sampling_probabilities))
        ...
        self.total_len = int(epoch_size)
```

均衡模式 `epoch_size = 源数 × 最大源长度`；加权模式 `epoch_size = max(len_i / p_i)`：

- 现状（拼接、不均衡）：183,168 → `steps/epoch = 1908`（batch=96）
- 5 任务等权：`5 × 64041 = 320,205` → `steps/epoch ≈ 3335`
- Press:Switch = 75:25：`max(len_i / p_i) = 256,164` → `steps/epoch ≈ 2668`

**`max_steps` 必须按新的 `steps/epoch` 重算**（否则实际 epoch 数会大幅偏离预期）。

---

## 10. 推理侧对齐

### 10.1 `task_descriptions` 只在 `inference` 段

训练**不读**它。训练 prompt 来自数据集每一帧的 `task_description`：

```349:350:fluxvla/datasets/parquet_dataset_v3.py
        data['task_description'] = self._resolve_task_description(
            dataset_idx, data)
```

```233:234:fluxvla/datasets/parquet_dataset_v3.py
        if isinstance(raw_task, (int, np.integer)):
            return self.tasks[dataset_idx].get(int(raw_task), '')
```

最终 prompt 模板：

```323:325:fluxvla/transforms/prompters.py
        full_prompt = f'Task: {cleaned_text}, State: {state_str}{suffix}'

        inputs['prompt'] = full_prompt
```
（`suffix = ';\nAction: '`，`cleaned_text` 只做 `strip()` / `_`→空格 / 去换行，**不做 lowercase**）

推理侧则从 `task_descriptions` 取：

```253:263:fluxvla/engines/runners/base_inference_runner.py
    def _get_task_description(self, task_id: str) -> str:
        ...
        return self.task_descriptions.get(
            task_id, 'place it in the brown paper bag with right arm')
```

```397:397:fluxvla/engines/runners/base_inference_runner.py
        obs['task_description'] = instruction
```

### 10.2 ⚠️ prompt 不匹配（已修复 2026-10-04，附实测代价）

**曾用的问题配置**：

```python
    task_descriptions={
        '1': 'complete the task',
    },
```

而训练 prompt 来自数据集，实际是 `Task: Press the red button, …` / `black` / `green`。
模型从没见过 `complete the task` 这句话 → prompt 分布偏移。

**实测代价**（`step-010000` 检查点，同一批 768 个样本、其余完全一致，只替换 prompt 里的任务文本）：

| | 正确 prompt | 错误 prompt | 变化 |
|---|---:|---:|---:|
| **总体 loss** | 0.00389 | **0.00598** | **+54%** |
| Press black | 0.00325 | 0.00537 | +65% |
| Press green | 0.00425 | 0.00720 | +69% |
| Press red | 0.00417 | 0.00537 | +29% |
| 左臂 0–6（几乎不动的维度） | 0.00031 | 0.00031 | ≈0 |
| **右臂 8–14（真正的动作维度）** | 0.00835 | **0.01313** | **+57%** |

右臂逐维误差上升 40~85%（`0.0069→0.0099`、`0.0128→0.0215` …），而**左臂几乎不变** —— 这个对照直接说明 prompt 影响的是「要执行什么动作」，不是噪声。

复现命令：`python pod_scripts/eval_tron2_per_task.py --prompt-override 'complete the task'`
（结果存档：`eval_step10000.json` / `eval_step10000_wrongprompt.json`）

**修法**（措辞与数据集里一字不差）：

```python
task_descriptions={
    '1': 'Press the red button',
    '2': 'Press the black button',
    '3': 'Press the green button',
    # 加 Switch 之后：
    # '4': 'Switch the selector from left to right',
    # '5': 'Switch the selector from right to left',
},
```

> `Tron2InferenceRunner` 自带的默认值是 `{'1': 'Complete the task.'}`（`tron2_inference_runner.py:83-85`），同样与训练不一致。
> 修这个**不需要重新训练** —— 它只影响推理入参，权重完全不变。

若希望部署时**用一句统一 prompt**（不按任务切换），那训练侧也必须统一改写任务文本（加 transform 或改 `tasks.parquet`），**这种情况需要重训**。

### 10.3 horizon / 执行步数

- `n_action_steps`（模型）应与训练的 `action_window_size` 相等
- `action_chunk`（执行步数）应是它的**子集**：多任务下建议取"最短任务一个完整动作单元"的长度，`32 / 50` 是当前取值
- RTC 场景可用 `execute_horizon` 覆盖 `action_chunk`（`base_inference_runner.py:377-379`）

---

## 11. 联合训练 vs 顺序训练

| 维度 | 联合（当前做法） | 顺序（先 A 后 B） |
|---|---|---|
| 最终产物 | 同一个模型 | 同一个模型（首尾相接的两次微调） |
| 遗忘风险 | ✅ 低 | ⚠️ 高 |
| 任务配比 | ✅ 可控 | ❌ 后训任务主导 |
| 每任务独立 horizon/统计 | 需 grouped / overrides / 多实例组织 | ✅ 天然分开 |
| batch 内跨任务混合 | ✅ 有 | ❌ 没有 |
| LR 调度 | 一条曲线 | ⚠️ 需处理（见下） |
| 训练时长 | 一次 | 两次之和（各需 warmup） |

### 顺序训练的三个坑

1. **统计必须锁死**：第二段必须沿用第一段的 `dataset_statistics_path`，不能重算
2. **LR 会被继承**：`--resume-from` 会恢复调度器状态
   ```512:521:fluxvla/engines/runners/base_train_runner.py
           # Restore scheduler state
           if ('scheduler_state_dict' in checkpoint_info
                   and self.lr_scheduler is not None):
               try:
                   self.lr_scheduler.load_state_dict(
                       checkpoint_info['scheduler_state_dict'])
   ```
   若第一段按 `decay_steps == max_steps` 全程退火，结束时 lr≈`min_lr`，第二段几乎学不动。
   处理：第一段用 `decay_steps ≫ max_steps`（半程退火），或改代码加 `--reset-scheduler`。
3. **LoRA 只有 resume 路径能恢复**：`_load_lora_adapter_state()` 只在 `_load_model_state()` 内被调用，而后者只在 `resume_from` 有值时执行。**不传 `--resume-from` 就等于从 base 重头开始。**
   ```170:196:fluxvla/engines/runners/ddp_train_runner.py
       def _resolve_lora_adapter_path(self) -> Optional[str]:
           """Locate the LoRA adapter saved next to the resumed checkpoint."""
           candidates = []
           if self.resume_from:
               ...
               candidates.append(
                   os.path.join(checkpoint_dir, f'{stem}-adapter.safetensors'))
               candidates.append(
                   os.path.join(run_dir, 'adapter_model.safetensors'))
   ```

**顺序训练命令示例**

```bash
# 第一段：数据 = A
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora_taskA.py \
  --work-dir work_dirs/taskA

# 第二段：数据 = B，从 A 的产物继续
torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora_taskB.py \
  --work-dir work_dirs/taskB \
  --resume-from work_dirs/taskA/checkpoints/latest-checkpoint.pt
```

---

## 12. 验证脚本

### 12.1 数据集差异体检

```bash
cd FluxVLA/datasets/RealRobot_Tron2_lerobot
/home/lab/miniconda3/envs/fluxvla/bin/python - <<'EOF'
import glob, os, json
import pyarrow.parquet as pq
import numpy as np
from collections import Counter

for root in sorted(glob.glob('lerobot_*')) + sorted(glob.glob('_archive/lerobot_*')):
    info = json.load(open(os.path.join(root, 'meta', 'info.json')))
    task = pq.read_table(os.path.join(root, 'meta', 'tasks.parquet')).to_pandas()['task'].tolist()
    L = []
    for f in glob.glob(os.path.join(root, 'meta', 'episodes', '**', '*.parquet'),
                       recursive=True):
        e = pq.read_table(f).to_pandas()
        L.extend(e['length'].tolist())
        suc = Counter(e['episode_success'].tolist()) if 'episode_success' in e.columns else None
    p = sorted(glob.glob(os.path.join(root, 'data', '**', '*.parquet'), recursive=True))[0]
    a = np.asarray(pq.read_table(p, columns=['action']).to_pandas()['action'].iloc[0])
    print(f"{root:44s} | {task[0][:34]:34s} | eps={len(L):4d} frames={sum(L):7d} "
          f"| ep_min/med/max={min(L)}/{int(np.median(L))}/{max(L)} | action_dim={a.shape[0]} "
          f"| fps={info.get('fps')} | success={suc}")
EOF
```

### 12.2 窗口内真实步数验证（改 horizon 后用）

```python
# 采样若干样本，确认窗口长度统一、有效步数符合预期
import collections
cnt = collections.Counter()
eff, shapes = [], collections.Counter()
for i in range(0, 200000, max(1, len(ds) // 300)):
    s = ds[i]
    shapes[tuple(s['actions'].shape)] += 1
    eff.append(float(s['action_masks'].sum()))
print('动作张量形状分布:', dict(shapes))   # 应只有 (action_horizon, 32) 一种
print('平均有效监督步数:', sum(eff) / len(eff))
```

### 12.3 确认统计没有"静默维度"

```python
import json, numpy as np
st = json.load(open('datasets/RealRobot_Tron2_lerobot/tron2_stats_X.json'))
for k in ('action', 'observation.state'):
    s = st['private'][k]
    lo, hi = np.asarray(s['q01']), np.asarray(s['q99'])
    span = hi - lo
    bad = np.where(span < 1e-3)[0]
    print(k, '跨度<1e-3 的维度:', bad.tolist(), '| 最小跨度:', span.min())
```

---

## 13. 常见错误速查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `np.stack` 报 shape 不一致 | 多数据源 `action_window_size` 不同，未启用 `PrepareStateActionTargets` | 启用并把 `action_horizon` 设为 `max(窗口)` |
| `horizon X exceeds target horizon Y` | `action_window_size > action_horizon` | 调大 `action_horizon`（只能补不能截） |
| `Automatic transformed statistics requires a common 'action_window_size'` | 多源窗口不一致还走自动统计 | 给 `dataset_statistics_path` 或用 grouped 按组配 |
| `grouped datasets should use grouped stats` | grouped 模式用了 `dataset_statistics_path` | 去掉该参数，按组配统计 |
| 训练正常但推理效果差、动作抖动 | 训练/推理 prompt 不一致（如 `'complete the task'`） | 对齐 `task_descriptions` 措辞 |
| 某任务几乎学不到 | 采样占比过低（如 Switch 只占 14.6%） | 换 balanced wrapper 或设 `sampling_weights`（见 9.6） |
| 总 loss 正常但某任务推理效果差 | loss 是整批混合的标量，小任务被稀释（见 9.8） | 加 per-task 指标监控；并提高该任务的采样或损失权重 |
| 某个关节维度的 loss 异常大 | 该维度在某任务里静止，统计跨度塌缩 | 人工修正 q01/q99（armsymmetric 方案）或分组统计 |
| 续训后 loss 不降 | `--resume-from` 继承了已退火到底的 lr | 第一段半程退火，或加 `--reset-scheduler` |
| 第二段训练把第一段"忘了" | 顺序训练的灾难性遗忘 | 混入少量 replay 数据，或改联合训练 |
| 换了数据集但 loss 从 0.006 跳回 0.5 | 统计变了 / 旧 checkpoint 不兼容 | 从头训 |
| 磁盘爆掉 | 每份 checkpoint ≈ 29.2 GB（`.pt` 14.07 + `.safetensors` 13.8 + adapter 0.14） | 调 `max_keep_ckpts` |

---

## 14. 关键代码位置索引

| 主题 | 位置 |
|---|---|
| 数据集三种组织格式 | `fluxvla/datasets/dataset_wrapper.py:39-45`、`104-217` |
| 均衡采样 wrapper | `fluxvla/datasets/balanced_dataset_wrapper.py:29-153` |
| 窗口构建 / 越界填充 / mask | `fluxvla/datasets/parquet_dataset_v3.py:261-355` |
| 任务文本解析 | `fluxvla/datasets/parquet_dataset_v3.py:219-234` |
| 多根拼接 | `fluxvla/datasets/parquet_dataset_v3.py:196-206` |
| `PrepareStateActionTargets` | `fluxvla/transforms/transform_inputs.py:222-335` |
| 归一化 + 维度补 0 | `fluxvla/transforms/normalize.py:670-853` |
| `RelativeActions`（state-relative） | `fluxvla/transforms/transform_actions.py:160-188` |
| prompt 模板 | `fluxvla/transforms/prompters.py:302-326` |
| 统计一致性检查 | `fluxvla/datasets/utils/transformed_statistics.py:514-535` |
| collator（等长约束） | `fluxvla/collators/dict_collator.py:51-59` |
| loss 的 action 维切片 | `fluxvla/models/vlas/pi0_flowmatching.py:260-272` |
| loss 的 mask 加权 | `fluxvla/engines/losses/rabc.py:38-63` |
| 推理固定 horizon | `fluxvla/models/vlas/pi0_flowmatching.py:844-846` |
| 推理 prompt 取值 | `fluxvla/engines/runners/base_inference_runner.py:253-263`、`397` |
| 执行步数 / RTC | `fluxvla/engines/runners/base_inference_runner.py:377-379`、`524-531` |
| LoRA adapter 恢复 | `fluxvla/engines/runners/ddp_train_runner.py:170-229` |
| 调度器状态恢复 | `fluxvla/engines/runners/base_train_runner.py:512-521` |
| 采样索引（决定任务占比） | `fluxvla/datasets/dataset_wrapper.py:600-614` |
| 均衡 / 加权采样 | `fluxvla/datasets/balanced_dataset_wrapper.py:29-153` |
| 损失归约（单标量 + mask 加权） | `fluxvla/models/vlas/pi0_flowmatching.py:756-766` |
| per-sample 权重通道 | `fluxvla/transforms/attach_rabc_weight.py:36-43`、`fluxvla/transforms/transform_inputs.py:208-210` |
| 本项目的训练配置 | `configs/pi05/pi05_paligemma_tron2_cabinet_lora.py` |
