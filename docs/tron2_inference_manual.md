# TRON2 柜内操作 — 推理部署手册

面向"已经训好权重、要上真机跑"的场景。所有命令都在开发机（`lab-MS-7E06`）上执行，
除相机 shim 那一节外。

**一句话流程**：起相机 shim → 起推理服务 → 离线自测 → `--goto-initial` 定位 → 预览 → `--execute` 执行。

---

## 0. 组件与数据流

```
相机模块 10.192.1.4:8770  ──HTTP JPEG──┐
                                        │
机器人本体 10.192.1.2:5000 ──WebSocket──┼─→ tron2_robot_client.py
                                        │        │  (观测打包)
                                        │        ↓
                                        │   ZMQ :5555 推理服务
                                        │        │  (50, 18) 绝对关节角
                                        │        ↓
                                        └──→ 安全层 → servoj @500Hz → 机器人
```

| 组件 | 位置 | 说明 |
|---|---|---|
| `camera_shim.py` | `10.192.1.4`（相机模块） | ROS2 → HTTP，纯订阅、只读 |
| `zmq_inference_server.py` | 本机 `:5555` | 加载权重，占 GPU 约 7.6 GB |
| `tron2_robot_client.py` | 本机 | 组装观测、调用推理、安全层、下发 |

---

## 1. 前置检查

```bash
# 机器人本体
timeout 3 bash -c "echo > /dev/tcp/10.192.1.2/5000" && echo "机器人 ✅" || echo "机器人 ❌"

# 相机 shim
curl -s -m 5 http://10.192.1.4:8770/status

# 推理服务
ss -ltn | grep 5555

# GPU 是否空着（训练会占满）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

> ⚠️ **推理服务和训练不能同时跑**。起推理服务前先确认训练已停。

---

## 2. 启动相机 shim（在 10.192.1.4 上）

shim 默认**不随开机自启**，重启后要手动拉起来；它是取图的唯一来源，没它就全链路不通。

```bash
sshpass -p 123456 ssh -o StrictHostKeyChecking=no guest@10.192.1.4 \
  'setsid bash -c "source /opt/ros/foxy/setup.bash && \
   exec python3 /home/guest/camera_shim.py --port 8770" \
   > /home/guest/camera_shim.log 2>&1 < /dev/null &'

# 验证：三路相机 age_ms 应 < 100
curl -s -m 5 http://10.192.1.4:8770/status
for c in top left right; do
  curl -s -m 5 -o /tmp/f_$c.jpg -w "  frame/$c HTTP %{http_code}\n" \
    http://10.192.1.4:8770/frame/$c
done
```

停止：
```bash
sshpass -p 123456 ssh guest@10.192.1.4 'pkill -f camera_shim.py'
```

---

## 3. 启动推理服务

```bash
cd /home/lab/tron_ws/FluxVLA
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled

setsid nohup /home/lab/miniconda3/envs/fluxvla/bin/python \
  scripts/zmq_inference_server.py \
  --config configs/pi05/pi05_paligemma_tron2_cabinet_lora_deploy.py \
  --ckpt-path work_dirs/tron2_buttons_v2/checkpoints/step-010000-epoch-005-loss=0.0041.safetensors \
  --port 5555 \
  > /home/lab/tron_ws/zmq_server.log 2>&1 < /dev/null &
```

首次加载约 **40 秒**（Triton JIT + CUDA Graph），之后稳态推理 **~52 ms**。

```bash
# 就绪判据
tail -20 /home/lab/tron_ws/zmq_server.log
ss -ltn | grep 5555

# 停止
pkill -f zmq_inference_server
```

---

## 4. 离线自测（**完全不碰机器人**）

用数据集里的真实观测（3 路 mp4 帧 + parquet 状态）走完整链路，验证服务、反归一化、
prompt 路由是否正常。**换权重/改配置后先跑这个。**

> ℹ️ TRON2 相关脚本都在 **`FluxVLA/scripts/`** 下（2026-10-05 从 workspace 根目录的
> `pod_scripts/` 合并进来，与 `train.py` / `tron2_robot_client.py` 同一层），
> 所以命令一律先 `cd /home/lab/tron_ws/FluxVLA`。

```bash
cd /home/lab/tron_ws/FluxVLA && unset PYTHONPATH
/home/lab/miniconda3/envs/fluxvla/bin/python \
  scripts/test_tron2_inference_offline.py --frame 60 --repeat 3
```

预期：

| 项 | 正常值 |
|---|---|
| 输出形状 | **(50, 18)** |
| finite | True |
| 服务端延迟 | ~52 ms |
| 右臂 MAE vs 数据集真值 | 0.008 ~ 0.025 rad |
| 换 prompt 后动作变化 | > 0（说明 prompt 在路由任务） |

---

## 5. 定位到起始位姿 `--goto-initial`

⚠️ **真实运动，必须在场、手放硬急停。**

```bash
cd /home/lab/tron_ws/FluxVLA && unset PYTHONPATH
/home/lab/miniconda3/envs/fluxvla/bin/python scripts/tron2_robot_client.py \
  --task 'Press the red button' --goto-initial --runs 1
```

- 会把双臂移到该任务数据集片段的**首帧均值**位姿，raised-cosine 平滑、默认 10 s。
- 不键入 `yes` 不会动；加 `--yes` 跳过确认。
- 只想定位、不跑推理循环：加 `--goto-stop`。
- 第一次建议先用小比例确认方向正确：`--goto-frac 0.1`。

> 三个数据集的首帧离散度是 **0.24~0.27 rad（约 15°）**，所以这是"近似起点"，
> 不是精确复现某一集。

---

## 6. 预览（**只读，不发任何指令**）

默认模式就是预览 —— 不加 `--execute` 时，机器人全程收不到任何指令。

```bash
/home/lab/miniconda3/envs/fluxvla/bin/python scripts/tron2_robot_client.py \
  --task 'Press the red button' --runs 3
```

输出末尾会明确写：`完成。提醒：本程序没有执行任何动作，机器人全程未收到指令。`

---

## 7. 执行 `--execute`

⚠️ **真实运动。手臂行为由策略决定，硬急停是唯一可靠的停止手段。**

```bash
/home/lab/miniconda3/envs/fluxvla/bin/python scripts/tron2_robot_client.py \
  --task 'Press the red button' --runs 8 --execute \
  --log-press /home/lab/tron_ws/press_red
```

不加 `--yes` 时会要求键入 `yes` 确认。

---

## 8. 任务文本（必须与数据集逐字一致）

三个任务**共用一个模型**，`task_description` 是唯一的区分手段。

```bash
--list-tasks     # 列出全部训练过的任务
```

| 任务文本 | 按钮中心 xyz（米，来自数据集） |
|---|---|
| `Press the red button` | `[0.6577, -0.2205, -0.3246]` |
| `Press the black button` | `[0.6616, -0.2180, -0.2683]` |
| `Press the green button` | `[0.6840, -0.1906, -0.2121]` |

**措辞、大小写一个字都不能改。** 训练侧只做 `strip()`、`_→空格`、去换行，
不做小写化。喂错的文本（如 `complete the task`）实测让 loss **+54%**。

按钮沿**竖直方向（z）**排列，相邻间隔 44~68 mm，**有效半径仅 10~15 mm**。

---

## 9. 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--task` | `Press the red button` | 任务文本，必须逐字一致 |
| `--runs` | 3 | 观测/推理/执行循环次数 |
| `--execute` | 关 | ⚠️ 真的下发；不写就是只读预览 |
| `--yes` | 关 | 跳过交互确认 |
| `--goto-initial` | 关 | ⚠️ 先移动到起始位姿 |
| `--goto-stop` | 关 | 定位完即退出 |
| `--save-dir` | 无 | 存观测/动作 npz |
| `--log-press` | 无 | 存末端位姿轨迹 + 打印 z 偏移 |
| `--send-gripper` | 关 | 下发右夹爪（拧旋钮任务需要；按钮任务不用） |
| `--exec-steps` | 50 | 每个 chunk 只执行前 K 步就重规划（K<50 = 提前重规划） |
| `--rtc-blend` | `none` | `exp` = 接缝处与上一段尾部做时序集成 |
| `--rtc-decay` | 重叠长度 | 混合过渡长度（步） |
| `--max-delta` | 0.10 rad | 相邻 servoj 指令最大增量 |
| `--max-reach` | 1.20 rad | 单 chunk 相对起点的最大包络 |
| `--chunk-dt` | 1/30 s | 一个动作步的时间 |
| `--engage-adaptive` | **开** | 斜坡长度按 gap 自适应（见下方"衔接斜坡"） |
| `--engage-time` | 0.4 s | 自适应开启时是**上限**；关闭时是固定值 |
| `--engage-min` | 0.05 s | 自适应下限（防止小 gap 变成力矩台阶） |
| `--engage-vel` | 1.0 rad/s | 过渡时的峰值参考速度上限 |
| `--samples` | 1 | 每 chunk 采样次数（>1 取平均降随机误差） |
| `--shim` / `--ws` / `--accid` | 见下 | 相机 / 机器人 / 序列号 |

默认值：`--shim http://10.192.1.4:8770`、`--ws ws://10.192.1.2:5000`、
`--accid DACH_TRON2A_215`、`--zmq tcp://127.0.0.1:5555`。

---

## 10. 输出怎么读

### 安全层四道闸（顺序执行）

```
左臂 7 维   : 已锁定为实测位姿（模型输出被丢弃）
步间限幅    : N 个动作值被 --max-delta 限住
包络限幅    : N 个动作值被 --max-reach 限住
绝对限位    : N 个动作值被手册限位限住
```

- **左臂恒定锁死**：5 个训练子集里左臂都不动，模型没有它的信号，一律丢弃模型输出。
- 若 `包络限幅` 长期不为 0，说明策略想走的路程超过包络，考虑（谨慎地）放大 `--max-reach`。

### 衔接斜坡（`--engage-time`）—— 它不该手调

从"发指令前实测位姿" `fresh` 到"计划第一步" `target` 之间要有一段升余弦过渡，
否则伺服会在**单个控制周期内**被要求完成全部修正（`tau ≈ kp*(q_ref - q_actual)`，
**kp 能到 420**）—— 就是进入伺服时那一下闷响。

**固定长度的问题是它必须按最坏情况配置**：只有 `--goto-initial` 之后的**第一段**
（模型加载期间机械臂耷拉着）gap 才大，后面每段 gap 都很小，
却都在为那个不存在的 gap 付 0.4 秒。

所以默认开启自适应：`T = pi*gap/(2*v_max)`，夹在 `[--engage-min, --engage-time]`。

| gap | 斜坡（自适应） | 固定 0.4 s 的浪费 |
|---:|---:|---:|
| 0.01 rad (0.6°) | **0.05 s** | 0.35 s |
| 0.05 rad (2.9°) | 0.08 s | 0.32 s |
| 0.10 rad (5.7°) | 0.16 s | 0.24 s |
| 0.20 rad (11.5°) | 0.31 s | 0.09 s |
| **>0.255 rad** | 0.40 s（被上限夹住） | 0 |

**调法：把 `--engage-time` 当「上限」用，可以放心开大**（如 0.8~1.0 s）——
正常段只会用 0.05~0.08 s，只有第一段那种大 gap 才会用到长斜坡。
`--engage-min 0.05` 和 `--engage-vel 1.0` 保持默认即可
（1.0 rad/s 对齐了计划自身的步速 0.6~1.35 rad/s）。

> ⚠️ gap 超过 **0.255 rad（14.6°）** 时会被上限夹住，此时实测峰值速度会
> **超过 `--engage-vel`**。如果你看到这个量级的 gap 还想要平滑过渡，就把
> `--engage-time` 调大而不是调小。
>
> 日志里两行可直接判断工作是否正常：
> `[执行] 位姿锚定: 推理后偏差 X.XXXX rad -> 用 Y.YY s 平滑衔接 (自适应，上限 Z)`
> 和 `[时间账] 衔接斜坡 Y.YY s 内实际位移 Z.ZZZZ rad`（位移≈0 会标注"等于原地停顿"）。

### 提前重规划（`--exec-steps` + `--rtc-blend`）

默认是"走完整段再重规划"：执行全部 50 步（1.667 s），然后才推理下一段。

`--exec-steps K`（K<50）改成"只执行前 K 步就重规划"，相邻两段因此**重叠 (50-K) 步** ——
上一段还留着一段对未来窗口的预测，正好和新段的覆盖区间重合。

`--rtc-blend exp` 把这段重叠用指数权重混合起来：

```python
w[i] = exp(-3 * i / (decay - 1))          # 与 rtc_guidance.compute_prefix_weights 同公式
blended[i] = w[i] * 上一段尾部[i] + (1 - w[i]) * 新段[i]
```

- **接缝处 w≈1**：贴近刚执行完的运动，保证连续
- **远离接缝 w→0**：让更晚、更新的观测接管
- 副带收益：两个独立预测取平均，把每次推理的随机抖动（实测 0.0106 rad）按 ~1/√2 削掉

```bash
python scripts/tron2_robot_client.py --task 'Press the red button' \
  --runs 8 --execute --exec-steps 30 --rtc-blend exp \
  --log-press /home/lab/tron_ws/press_red
```

**安全**：混合会改变轨迹，所以混合之后**会重新过一次 `--max-delta` 步间限幅**
（`clamp_step_delta`）—— 两个各自合法的计划混合后仍可能出现步间跳变，那道闸必须再走一遍。
日志里会打印被限住的动作值个数。

**注意**：
- `--exec-steps` 必须 < 50 才产生重叠；`=50` 时本功能自动跳过并在日志里说明
- K 越小重规划越频繁 → 推理耗时（52 ms）占比越高。K=30 时约为 5%
- `--engage-time`（默认 0.4 s）是每次重规划时的进入斜坡，重规划变频繁时它会占掉周期的大头，
  **建议同步降到 0.15 s 左右**

#### ⚠️ 实测结论：这套机制在当前配置下**不划算**（2026-10-05）

三组对照，red 按钮、各 20 段，指标**只看 z**（x/y 是机位差异，见 §11）：

| 配置 | z 均值 | z 标准差 | 最近接近中位 |
|---|---:|---:|---:|
| **K=50（完整执行）** | **+20.0 mm** | **±6.9 mm** | 35.0 mm |
| K=40 | +27.9 mm | ±9.2 mm | 34.9 mm |
| K=40 + `--rtc-blend exp` + 自适应斜坡 | +32.6 mm | ±12.4 mm | 44.2 mm |
| 原始基线（K=50，8 段） | +21.9 mm | ±6.7 mm | 40.7 mm |

**缩短视界是主要代价来源（+7.9 mm），blend 与自适应斜坡再叠加 +4.7 mm。**

原因是模型那 50 步是"从**一次**观测出发的完整专家轨迹"。只执行前 K 步，等于在动作
做到一半时强行打断、用一次新预测重做剩下的部分 —— **每次中途重新决策都会引入误差**。
模型自己的完整计划，比任何重新规划的版本都好。

**结论：保持 `--exec-steps 50`、不启用 `--rtc-blend`。**
这套机制保留在代码里（将来若改用更长的 chunk 或真正的异步推理方案可参考），
但以当前模型和任务，开启即掉精度。

### 时间预算

每个 chunk 执行 **50 步 × 1/30 s = 1.667 s**（`n_action_steps=50`）。
按 500 Hz 展开成 `(50-1) × 17 = 833` 条 servoj 指令。
推理 ~52 ms 夹在两次执行之间，占比约 **3%**。

> ⚠️ 训练配置里还有一个 `action_chunk=32`（`pi05_paligemma_tron2_cabinet_lora.py:371`），
> 但它**只被 `fluxbisim_base_inference_runner` 使用**（`denormalized[:self.action_chunk]`）。
> **ZMQ 部署路径和客户端都不截断**，客户端拿到的是完整 50 步。

---

## 11. 按压精度怎么判 —— **只看 z**

这是本手册最容易被搞错的一条。

数据集是**在多个机位录制**的，`--goto-initial` 之后机器人在哪一站，x/y 就整体平移到哪里。
实测证据：按压点 y 偏移能被起始位姿 y 偏移预测（相关 **0.78~0.88**），而 z 与起始位姿
相关≈0、1513 集里始终只有 **3.3~4.7 mm** 离散。

| 轴 | 含义 | 能否当误差 |
|---|---|---|
| **z** | 按钮排列方向 = **决定按哪个按钮** | ✅ 唯一的精度指标 |
| x | 前伸深度 | ❌ 随机位平移 |
| y | 沿面板横移 | ❌ 随机位平移 |

**所以不要拿"离按钮中心的 3D 距离"判定成败** —— 那会把机位差异算成误差。

`--log-press` 的输出只报 z 作为精度：

```
z 偏移（按钮排列方向 = 决定按哪个按钮）:
  均值 +X.X mm   标准差 ±X.X mm   范围 ...
判据：按钮有效半径 10~15 mm；演示数据自身在 z 上做到了 3~5 mm，
      所以 |均值| 和 标准差 都要 <= 5 mm 才算追平演示
```

参考基线（实机闭环 8 次/任务）：

| 任务 | 实机 z 偏移 | 演示 z 精度 | 结论 |
|---|---:|---:|---|
| black | +2.5 mm | 4.5 mm | ✅ 达演示水平 |
| green | -3.2 mm | 4.7 mm | ✅ 达演示水平 |
| red | +21.9 mm | 3.3 mm | ❌ 明显偏（往 black 一侧） |

---

## 12. 已知事实（别再踩）

1. **返回的 18 维是【绝对关节角】，不是 delta。**
   `DenormalizeDeltaAction` 已在 `normalize.py:507` 做了 `action[..., :dims] += state`，
   delta 被还原过了。别再拿它减当前位姿 —— 那样会得到荒谬的值。

2. **左夹爪不是严格 0**，实测 ±0.0039（离线）/ ±0.0156（真机）。量级 1e-2，实践上≈0，
   但不是 bit-exact。文档里"恒为 0.0000"的说法不准。

3. **头部不在数据集的 16 维状态里**，`--goto-initial` 也不管它。
   但头部会改变 `cam_high` 的取景。采集时的头部值可从**原始 21 维数据**（`datasets_raw/`）
   的第 16/17 维读到，为恒定值 **`[0.5663, 0.0057]`**。

4. **三路相机的重要性极不均衡**（消融实测）：
   - `cam_right_wrist`：换成真机图 → 右臂 MAE **+330%**，**绝对主导**
   - `cam_high`：±90 px 平移 → MAE 只动 ±5%；换成真机图 +5% → **基本无关**
   - `cam_left_wrist`：扰动它甚至会让指标变好 → **纯噪声**
   排查精度问题时，把精力放在右腕相机上。

5. **机器每 5 天在 03:00 自动重启**（`/usr/local/bin/reboot-every-5days.sh`），
   长任务要能断点续跑。

6. **机器人状态查询必须复用 WebSocket，否则每段之间会多出约 1 秒停顿。**
   实测：新建一条 WebSocket 要 **~330 ms**，而**查询本身只要 ~1 ms**（两种请求都一样）：
   ```
   read_robot_state (新连接) : 334 / 339 ms
   read_ee_pose     (新连接) : 329 / 333 / 334 ms
   复用连接连续 6 次         : 1 ms × 6
   ```
   客户端一轮过去要开 3 条（观测状态、末端诊断、开流前 fresh），加上 `--log-press`
   的监控线程还有第 4 条 —— 白白烧掉约 1 秒。现已统一走 `RobotQueryLink`（一条长连接 + 锁 + 断线重连），
   观测耗时从 **~500 ms 降到 14~24 ms**。

   > 330 ms 对局域网握手来说异常偏高（正常应 <5 ms）。观察到 `dmesg` 里
   > `wlo1` 在持续 deauth/reassociate，可能与此有关 —— 复用连接后不再影响主循环，
   > 但其他频繁建连的地方值得留意。

---

## 13. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `camera shim 不可用` | shim 没跑（重启后必现） | 见 §2 |
| 相机 `age_ms` 很大 | shim 卡住 | 重启 shim，看 `~/camera_shim.log` |
| `ZMQ 连接超时` | 推理服务没起 | 见 §3 |
| `CUDA out of memory` | 训练占着显存 | `pkill -f scripts/train.py`，确认 `nvidia-smi` 干净再起 |
| 输出全 0 或 `⚠NaN` | 权重/归一化对不上 | 换回已知可用检查点；跑 §4 自测 |
| 任务文本报错 | 措辞与数据集不一致 | 用 `--list-tasks` 核对 |
| 机器人不动 | 保险急停 / 未进伺服 | 看 servoj 回包里的 `notify_servoJ`（`fail_motor` 表示电机故障） |
| 动作被大量限幅 | 当前位姿离数据集起点太远 | 先 `--goto-initial` |

---

## 14. 停止与清理

```bash
pkill -f tron2_robot_client                 # 停客户端
pkill -f zmq_inference_server               # 停推理服务（释放 ~7.6 GB 显存）
sshpass -p 123456 ssh guest@10.192.1.4 'pkill -f camera_shim.py'   # 停相机 shim
```

机器人会**保持最后一条 servoj 指令的位置**，不会自动归位。

---

## 15. 训练侧衔接

- 续训命令与断点见 `docs/pi05_tron2_cabinet_lora_training_config.md` 第 13 章。
  续训必须用 `.pt`（含优化器/调度器状态），**不是** `.safetensors`。
- 续训正确性判据：启动后第一个 `Global Step` 的 loss 应接在中断处
  （step 10000 时约 **0.0036**）。若跳到 **0.39**，说明 LoRA 权重被静默丢弃了（历史 bug）。
- `scripts/train_tron2_buttons.sh` **不支持 `--resume-from`**，续训要直接跑 `torchrun`。
