#!/usr/bin/env python3
"""TRON2 cabinet policy: robot-side observation loop (READ ONLY).

Builds a real observation from the live robot and asks the local ZMQ inference
server for an action chunk.  It **never commands the robot** -- there is no
motion code path in this file at all, by design.  Use it to (a) prove the link
robot -> inference server works end to end and (b) sanity check the predicted
actions against the current pose before anyone enables motion.

Data sources (both reachable from the inference machine):

  cameras   http://<dev-module>:8770/frame/{top,left,right}
            `camera_shim.py` on the dev module (10.192.1.4), which subscribes to
            the ROS2 compressed topics.  Start it with:
                source /opt/ros/foxy/setup.bash && python3 ~/camera_shim.py --port 8770

  joints    ws://<motion-computer>:5000  `request_get_joint_state`
            returns q[16] in /joint_states order:
                [L7, R7, head_pitch, head_yaw]
            Grippers are NOT in this vector (they live on /gripper_state) and are
            deliberately pinned to 0.0 here -- see LOCK_GRIPPERS below.

Layout mapping (verified against the data platform export, 21 dims, first 16 kept):
    platform: [abad_L..wrist_roll_L, left_gripper, abad_R..wrist_roll_R,
               right_gripper, head_pitch, head_yaw, linear_x, angular_z, lifter]
    dataset : [L7, gripL, R7, gripR]                      <- model order, 16 dims
    robot   : [L7, R7, head_pitch, head_yaw, gripL, gripR] <- native order, 18 dims

Color note: the compressed topics publish bgr8 JPEG, so cv2 decoding yields BGR,
while the training videos decode to RGB.  We swap, otherwise red/blue are
interchanged relative to training.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import threading
import time
import uuid

import numpy as np

REPO = os.environ.get('FLUXVLA_ROOT', '/home/lab/tron_ws/FluxVLA')
sys.path.insert(0, REPO)

CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
SHIM_SLOT = {
    'cam_high': 'top',
    'cam_left_wrist': 'left',
    'cam_right_wrist': 'right',
}
# Gripper units: the robot reports 0-100 while the dataset stores 0-1, so the
# client divides by 100.  Confirmed three ways -- the official SDK doc's
# section 4.3 (left_opening 0-100), the repo's Tron2Operator docstring
# ("hw 0-100, we /100 -> 0-1") and its get_latest_gripper_state(), and the live
# reading (right_opening 2 -> 0.02, i.e. almost closed).
GRIPPER_SCALE = 100.0
READ_GRIPPER = True
GRIPPER_FALLBACK = 0.0
HEAD_PIN = 0.0  # only used for the 18-dim `states` vector

# The five training tasks and how they use the gripper (measured per subset):
#   Press the red/black/green button       gripR 0.00 -> 0.00   (never moves)
#   Switch the selector left<->right       gripR 0.01 -> 0.94   (grasps the knob)
# The left gripper is 0.0 across all five subsets, so it stays locked.
# Sending the right gripper is therefore required for the two selector tasks and
# is opt-in via --send-gripper; with the flag off this client only reads.
LEFT_GRIPPER_DEGENERATE = True

# --------------------------------------------------------------------------- #
# Safety layer
# --------------------------------------------------------------------------- #
# Joint limits from the TRON2 user manual section 1.5 (the manual says these are
# mandatory soft limits).  Order is the /joint_states order, which we verified
# equals the manual's table order: two joints in the recorded training data sit
# within a few milliradians of these bounds (wrist_yaw_R -1.394 vs -1.39,
# wrist_pitch_R -0.786 vs -0.78), so the index alignment is right even though
# the manual names joints proximal_pitch/roll/yaw while the robot reports
# abad/hip/yaw/knee.
JOINT_LIMITS = np.array([
    [-3.14, 2.60],    # 0  left  shoulder 1
    [-0.26, 3.19],    # 1  left  shoulder 2
    [-3.66, 1.48],    # 2  left  shoulder 3
    [-2.61, 0.26],    # 3  left  elbow
    [-1.74, 1.39],    # 4  left  wrist yaw
    [-0.78, 0.78],    # 5  left  wrist pitch
    [-1.57, 1.57],    # 6  left  wrist roll
    [-3.14, 2.60],    # 7  right shoulder 1
    [-3.19, 0.26],    # 8  right shoulder 2
    [-1.48, 3.66],    # 9  right shoulder 3
    [-2.61, 0.26],    # 10 right elbow
    [-1.39, 1.74],    # 11 right wrist yaw
    [-0.78, 0.78],    # 12 right wrist pitch
    [-1.57, 1.57],    # 13 right wrist roll
    [-0.78, 1.04],    # 14 head pitch
    [-1.57, 1.57],    # 15 head yaw
], dtype=np.float64)

# Rate limit: the largest jump allowed *between two consecutive servo
# commands*, the first command being measured against the pose the robot
# reports right now.  Bounding the increment rather than the offset from the
# start pose is what lets a chunk actually travel; an offset bound pins every
# step of the chunk inside one small box around the start and the arm crawls.
DEFAULT_MAX_DELTA = 0.10          # rad, ~5.7 deg per servo step
DEFAULT_HEAD_MAX_DELTA = 0.05     # rad; the head is cosmetic and slow

# Envelope: how far one chunk may end up from the pose it started at.  This is
# the naming-independent protection -- it caps how much can go wrong if a joint
# is mislabelled, without constraining the shape of a legitimate trajectory.
DEFAULT_MAX_REACH = 1.00          # rad, ~57 deg per chunk
DEFAULT_HEAD_MAX_REACH = 0.20     # rad


def plan_commands(actions: np.ndarray, states18: np.ndarray,
                  max_delta: float = DEFAULT_MAX_DELTA,
                  max_reach: float = DEFAULT_MAX_REACH,
                  lock_left_arm: bool = True):
    """Turn a predicted chunk into the joint commands that may be executed.

    Applies, in order: the left-arm lock, the absolute joint limits (never
    narrowed below the measured pose, so the clamp can never fight where the
    robot already is), the per-chunk reach envelope, then the per-step rate
    limit.  The last two are ordered so that the rate limit cannot push a
    command back outside the envelope.

    Returns ``(plan, info)`` where ``plan`` is [T, 16] in servoj order
    ``[L7, R7, head2]`` and ``info`` records what was overridden.
    """
    cur = np.concatenate([states18[0:7], states18[7:14], states18[14:16]])
    cmd = np.concatenate([actions[:, 0:7], actions[:, 7:14], actions[:, 14:16]],
                         axis=1).astype(np.float64)
    info = {'left_locked': 0, 'abs_clamped': 0, 'rate_clamped': 0,
            'reach_clamped': 0}

    if lock_left_arm:
        # The left arm is constant across all five training subsets, so the
        # policy has no signal for it.  Hold the measured pose instead of
        # executing whatever the model guessed.
        if not np.allclose(cmd[:, :7], cur[:7], atol=1e-9):
            info['left_locked'] = int(np.any(
                ~np.isclose(cmd[:, :7], cur[:7], atol=1e-6)))
        cmd[:, :7] = cur[:7]

    lo = JOINT_LIMITS[:, 0].copy()
    hi = JOINT_LIMITS[:, 1].copy()
    lo = np.minimum(lo, cur)
    hi = np.maximum(hi, cur)

    before = cmd.copy()
    cmd = np.clip(cmd, lo, hi)
    info['abs_clamped'] = int(np.count_nonzero(~np.isclose(before, cmd)))

    reach = np.full(16, np.inf if max_reach is None else max_reach)
    reach[14:16] = DEFAULT_HEAD_MAX_REACH
    before = cmd.copy()
    cmd = np.clip(cmd, cur - reach, cur + reach)
    info['reach_clamped'] = int(np.count_nonzero(~np.isclose(before, cmd)))

    delta = np.full(16, max_delta)
    delta[14:16] = DEFAULT_HEAD_MAX_DELTA
    before = cmd.copy()
    prev = cur
    for t in range(cmd.shape[0]):
        cmd[t] = np.clip(cmd[t], prev - delta, prev + delta)
        prev = cmd[t]
    info['rate_clamped'] = int(np.count_nonzero(~np.isclose(before, cmd)))

    return cmd.astype(np.float32), info


def ws_request(ws, accid: str, title: str, data: dict, timeout: float = 6.0):
    """Send one request and wait for its matching response (by guid)."""
    guid = uuid.uuid4().hex
    ws.send(json.dumps({
        'accid': accid,
        'title': title,
        'timestamp': int(time.time() * 1000),
        'guid': guid,
        'data': data or {},
    }))
    ws.settimeout(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = json.loads(ws.recv())
        except Exception:                # noqa: BLE001
            break
        if msg.get('guid') != guid:
            continue
        return msg.get('data', {})
    return {'result': 'no_response'}


def ws_send_raw(ws, accid: str, title: str, data: dict):
    """Send one request without waiting for the reply (fire and forget).

    The deployed policy streams ``request_servoj`` at 500 Hz and never blocks on
    the answer (``Tron2Operator._ws_send_request``), so a high-rate servo test
    must not wait either: a round trip per step would cap the rate far below the
    cadence the controller expects.
    """
    ws.send(json.dumps({
        'accid': accid,
        'title': title,
        'timestamp': int(time.time() * 1000),
        'guid': uuid.uuid4().hex,
        'data': data or {},
    }))


def ws_drain(ws, seconds: float):
    """Collect whatever the robot pushed back during ``seconds``.

    Used after a fire-and-forget stream to catch error replies that arrive once
    nobody is waiting for them.
    """
    out = []
    ws.settimeout(seconds)
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            out.append(json.loads(ws.recv()))
        except Exception:                # noqa: BLE001
            break
    return out


# --------------------------------------------------------------------------- #
# cameras
# --------------------------------------------------------------------------- #
def fetch_frames(shim_base: str, timeout: float = 5.0):
    import cv2
    import urllib.request

    out = {}
    for key in CAMERAS:
        url = '%s/frame/%s' % (shim_base.rstrip('/'), SHIM_SLOT[key])
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            buf = np.frombuffer(resp.read(), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError('failed to decode %s' % url)
        # bgr8 JPEG -> swap so the model sees RGB like the training videos
        out[key] = np.ascontiguousarray(bgr[:, :, ::-1])
    return out


def shim_status(shim_base: str):
    import urllib.request
    try:
        with urllib.request.urlopen(
                '%s/status' % shim_base.rstrip('/'), timeout=5) as r:
            return json.loads(r.read())
    except Exception as exc:            # noqa: BLE001
        return {'error': str(exc)}


# --------------------------------------------------------------------------- #
# joints
# --------------------------------------------------------------------------- #
def read_robot_state(ws_url: str, accid: str, timeout: float = 6.0):
    """Read joints (16) and gripper openings (0-1) over the robot WebSocket."""
    import websocket

    ws = websocket.create_connection(ws_url, timeout=timeout)
    try:
        js16 = _request(ws, accid, 'request_get_joint_state', timeout)['q']
        grip = [GRIPPER_FALLBACK, GRIPPER_FALLBACK]
        if READ_GRIPPER:
            try:
                g = _request(ws, accid, 'request_get_limx_2fclaw_state', timeout)
                grip = [float(g['left_opening']) / GRIPPER_SCALE,
                        float(g['right_opening']) / GRIPPER_SCALE]
            except Exception as exc:     # noqa: BLE001
                print('  [warn] 夹爪状态读取失败，回退 %.1f: %s'
                      % (GRIPPER_FALLBACK, exc))
        return np.asarray(js16, dtype=np.float32), np.asarray(grip, dtype=np.float32)
    finally:
        ws.close()


def _request(ws, accid: str, title: str, timeout: float):
    guid = uuid.uuid4().hex
    ws.send(json.dumps({
        'accid': accid,
        'title': title,
        'timestamp': int(time.time() * 1000),
        'guid': guid,
        'data': {},
    }))
    ws.settimeout(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = json.loads(ws.recv())
        except Exception:                # noqa: BLE001
            break
        if msg.get('guid') != guid:
            continue
        data = msg.get('data', {})
        if data.get('result') != 'success':
            raise RuntimeError('%s -> %s' % (title, data.get('result')))
        return data
    raise RuntimeError('%s timed out' % title)


def build_observation(js16: np.ndarray, grip2: np.ndarray):
    """Turn robot joints + grippers into the two vectors the server wants."""
    if js16.shape[-1] != 16:
        raise ValueError('expected 16 joint values, got %d' % js16.shape[-1])

    grip_l = float(grip2[0])
    grip_r = float(grip2[1])

    # 16 dims, model order [L7, gripL, R7, gripR] -> normalized against the
    # 16-dim quantile statistics, then padded to 32 by the dataset pipeline.
    qpos = np.concatenate([
        js16[0:7],
        [grip_l],
        js16[7:14],
        [grip_r],
    ]).astype(np.float32)

    # 18 dims, robot order [L7, R7, head(2), gripL, gripR] -> delta base and head.
    states = np.concatenate([
        js16[0:7],
        js16[7:14],
        js16[14:16],
        [grip_l, grip_r],
    ]).astype(np.float32)
    return qpos, states


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def predict(obs: dict, zmq_addr: str, unnorm_key: str, seed: int,
            episode_id: str = 'readonly-loop', reset: bool = True):
    import zmq
    from fluxvla.engines.runners.serving.serializers import MsgSerializer

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 600_000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(zmq_addr)
    try:
        sock.send(MsgSerializer.to_bytes({
            'endpoint': 'predict_action',
            'data': {
                'observation': obs,
                'unnorm_key': unnorm_key,
                'episode_id': episode_id,
                'seed': seed,
                'reset': reset,
                'request_id': uuid.uuid4().hex,
            },
        }))
        resp = MsgSerializer.from_bytes(sock.recv())
    finally:
        sock.close()
        ctx.term()

    if not resp.get('ok', False):
        raise RuntimeError('server error: %s' % resp.get('error'))
    return (np.asarray(resp['actions'], dtype=np.float32),
            float(resp.get('inference_time_s', 0.0)))


def trained_tasks():
    """Map every training task string to the subset it came from.

    The single policy was trained on all five tasks, and the only thing that
    tells it which one to perform is `task_description` -- so the operator has
    to pick the right string at run time.
    """
    pattern = os.path.join(
        REPO, 'datasets/RealRobot_Tron2_lerobot/lerobot_*/meta/tasks.parquet')
    out = {}
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(pattern)):
        try:
            tbl = pq.read_table(path)
            out[path.split('/')[-3]] = str(tbl['task'].to_pylist()[0])
        except Exception:                # noqa: BLE001
            continue
    return out


def resolve_task(requested: str, tasks: dict) -> str:
    """Exact match, then case-insensitive match, otherwise list what is valid."""
    if requested in tasks.values():
        return requested
    lowered = {v.lower(): v for v in tasks.values()}
    if requested.lower() in lowered:
        return lowered[requested.lower()]
    raise SystemExit(
        '未知任务 %r。训练过的任务只有：\n%s\n'
        % (requested, '\n'.join('  %-40s  (来自 %s)' % (v, k)
                                for k, v in sorted(tasks.items(),
                                                   key=lambda kv: kv[1]))))


def initial_pose_for_task(subset: str):
    """Mean first-frame arm pose across one subset's episodes.

    Returns ``(arm14, spread, n)``: the 14 arm joints in servoj order
    ``[L7, R7]``, the largest per-joint standard deviation across the episodes'
    first frames (how reproducible that start pose actually is), and the number
    of episodes averaged.

    The recorded state is 16-dim ``[L7, gripL, R7, gripR]``, so the grippers at
    index 7 and 15 are dropped.  The head is not recorded at all and is left
    where it is.
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(
        REPO, 'datasets/RealRobot_Tron2_lerobot', subset,
        'data/chunk-*/file-*.parquet')))
    if not files:
        raise RuntimeError('no parquet under %s' % subset)
    firsts = []
    for path in files:
        table = pq.read_table(path, columns=['observation.state'])
        firsts.append(np.asarray(table['observation.state'].to_pylist(),
                                 dtype=np.float64)[0])
    stack = np.asarray(firsts)
    mean = stack.mean(axis=0)
    return np.concatenate([mean[0:7], mean[8:15]]), stack.std(axis=0).max(), \
        len(files)


def ease_to_pose(cur16: np.ndarray, target14: np.ndarray,
                 seconds: float, hz: int) -> np.ndarray:
    """Ease the 14 arm joints from ``cur16`` to ``target14`` over ``seconds``.

    Raised-cosine in time, so the reference has zero velocity at both ends and
    no step to jerk the controller.  Head entries of the 16-dim command are
    held at the measured pose.
    """
    steps = max(int(seconds * hz), 2)
    u = np.linspace(0.0, 1.0, steps)
    gain = 0.5 - 0.5 * np.cos(np.pi * u)
    traj = np.tile(np.asarray(cur16, dtype=np.float64), (steps, 1))
    for i in range(14):
        traj[:, i] = cur16[i] + (target14[i] - cur16[i]) * gain
    return traj


def gripper_command(act_grip01: float) -> float:
    """Model gripper action (0-1) -> the robot's 0-100 opening command."""
    return float(np.clip(act_grip01, 0.0, 1.0)) * GRIPPER_SCALE


# ServoJ gains copied from fluxvla/engines/operators/tron2_operator.py, whose
# order is abad, hip, yaw, knee, wrist_yaw, wrist_pitch, wrist_roll per arm.
SERVOJ_KP = [420, 420, 300, 300, 200, 200, 200,
             420, 420, 300, 300, 200, 200, 200, 60, 60]
SERVOJ_KD = [12, 12, 15, 15, 10, 10, 10,
             12, 12, 15, 15, 10, 10, 10, 3, 3]

# The official protocol (SDK guide 3.6.4.1) documents request_servoj as
#     {"filter_ratio": <0..1>, "q": [16]}
# where 1.0 means "trust the reference completely, no filtering".
#
# Measured on robot-tron2-r-2.1.24: leave this field out and the robot answers
# response_servoj {"result": "success"} for *every* message and then ignores it
# outright -- a stream commanding +0.15 rad on head_yaw produced 0.0000 rad of
# motion.  The identical stream with filter_ratio present drove the joint to
# 0.1498 rad, i.e. it tracked.  A success reply therefore proves nothing; only
# the encoders do.
#
# The v/kp/kd/tau/mode/na fields below come from Tron2Operator and are accepted
# but ignored by this firmware -- they are kept so the message stays comparable
# with what the deployed operator sends.
SERVOJ_FILTER_RATIO = 1.0

# The deployed policy streams servoj at 500 Hz ("minimum for stable control" in
# tron2_operator.py) and paces on an absolute time axis instead of waiting for
# replies.  A lone servoj message therefore does not look like what the
# controller normally sees, so the servo self-test replays a short burst of the
# real shape -- same message, same cadence, target held at the measured pose.
SERVOJ_STREAM_HZ = 500
SERVOJ_STREAM_SECONDS = 1.0

# Time between two action steps.  The TRON2 recordings are 30 fps and the TRON2
# runner sets publish_rate=30 (see
# configs/gr00tn15/gr00tn15_eagle_3b_tron2_3cam_full_finetune.py), so one step
# spans 1/30 s and the 50-step chunk the deploy config asks for lasts 1.667 s.
DEFAULT_CHUNK_DT = 1.0 / 30.0

# Joints used by the "wiggle" self-test: the least risky motion that still
# proves servoj can actually drive the arm.  Index 15 is head_yaw (cosmetic
# only) and index 6 is wrist_roll_L (spins the wrist about its own axis), in
# the /joint_states order documented at the top of this file.
WIGGLE_JOINTS = [
    (15, 0.10, 'head_yaw  idx15（头部偏转，装饰性，风险最低）'),
    (6, 0.05, 'wrist_roll_L  idx6（腕部绕自身轴自转）'),
]
WIGGLE_SECONDS = 2.0


def out_and_back(cur: np.ndarray, idx: int, amp: float,
                 steps: int) -> np.ndarray:
    """Build a [steps, 16] trajectory that eases 0 -> amp -> 0 on one joint.

    Every other joint is held at the pose the robot already reports, so only
    ``idx`` is asked to move.  The raised-cosine gain has zero slope at both
    ends, which keeps the servo reference free of velocity steps.
    """
    u = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    gain = 0.5 - 0.5 * np.cos(2.0 * np.pi * u)
    traj = np.tile(np.asarray(cur, dtype=np.float64)[:16], (steps, 1))
    traj[:, idx] = cur[idx] + amp * gain
    return traj


def stream_trajectory(ws, accid: str, traj: np.ndarray, hz: int):
    """Push a [T, 16] trajectory as a servoj stream.

    Returns ``(elapsed_seconds, stats)`` where ``stats`` counts the robot's
    replies and how many of them were not ``success``.

    Sends fire-and-forget on an absolute time axis, exactly like
    ``Tron2Operator._servo_step`` at 500 Hz, so the cadence the robot sees here
    is the one it sees in deployment.

    Replies are consumed by a background thread.  The robot answers every
    servoj, and at 500 Hz an undrained receive buffer fills within a couple of
    seconds -- after which the robot's own sends block and every later request
    on that socket times out.  The deployed operator is saved from this only
    because its WebSocketApp callback drains continuously.
    """
    steps = traj.shape[0]
    state = {'stop': False, 'replies': 0, 'failed': 0}

    def drainer():
        ws.settimeout(0.3)
        while not state['stop']:
            try:
                msg = json.loads(ws.recv())
            except Exception:            # noqa: BLE001
                continue
            if str(msg.get('title', '')).startswith('response_'):
                state['replies'] += 1
                if msg.get('data', {}).get('result') != 'success':
                    state['failed'] += 1

    reader = threading.Thread(target=drainer, daemon=True)
    reader.start()

    t0 = time.perf_counter()
    for i in range(steps):
        ws_send_raw(ws, accid, 'request_servoj', {
            'filter_ratio': SERVOJ_FILTER_RATIO,
            'q': [float(v) for v in traj[i]],
            'v': [0.0] * 16,
            'kp': SERVOJ_KP[:16],
            'kd': SERVOJ_KD[:16],
            'tau': [0.0] * 16,
            'mode': [0] * 16,
            'na': 0,
        })
        target = t0 + (i + 1) / hz
        remaining = target - time.perf_counter()
        if remaining > 1e-3:
            time.sleep(remaining - 1e-3)
        while time.perf_counter() < target:
            pass
    elapsed = time.perf_counter() - t0

    state['stop'] = True
    reader.join(timeout=1.0)
    return elapsed, state


def interpolate_plan(plan: np.ndarray, dt: float, hz: int) -> np.ndarray:
    """Expand [T, 16] waypoints into the >=500 Hz reference the manual wants.

    Mirrors ``Tron2Operator._run_trajectory_servoj``: each consecutive waypoint
    pair is linearly interpolated over ``dt`` seconds, so the controller sees
    the same message shape and cadence it sees in deployment.  With the TRON2
    settings (dt = 0.1 s, hz = 500) that is 50 sub-steps per segment.
    """
    sub = max(int(round(dt * hz)), 1)
    rows = []
    for seg in range(plan.shape[0] - 1):
        q0 = np.asarray(plan[seg], dtype=np.float64)
        q1 = np.asarray(plan[seg + 1], dtype=np.float64)
        for k in range(1, sub + 1):
            rows.append(q0 + (k / sub) * (q1 - q0))
    return np.asarray(rows, dtype=np.float64)


def run_wiggle(ws, accid: str, js, ws_url: str, timeout: float = 6.0) -> int:
    """Drive one joint out and back with servoj, watching the encoders follow.

    This answers "can servoj actually move the arm" -- every earlier self-test
    only ever commanded the pose the robot already held.  Joint states are read
    on a *second* WebSocket connection, because reading on the servo socket
    would stall the 500 Hz stream the controller expects.
    """
    import websocket

    base = np.asarray(js, dtype=np.float64)[:16]
    hz = SERVOJ_STREAM_HZ
    steps = int(hz * WIGGLE_SECONDS)

    print('>> wiggle: 逐关节 出去再回来')
    print('   %d Hz x %.1fs = %d 条/关节；关节采样走第 2 条连接（不打断节拍）'
          % (hz, WIGGLE_SECONDS, steps))
    print()

    for idx, amp, label in WIGGLE_JOINTS:
        lo, hi = JOINT_LIMITS[idx]
        if not (lo <= base[idx] - amp and base[idx] + amp <= hi):
            print('--- %s 跳过: ±%.2f 会超出限位 [%.2f, %.2f]'
                  % (label, amp, lo, hi))
            print()
            continue

        traj = out_and_back(base, idx, amp, steps)
        peak = int(np.argmax(traj[:, idx]))
        print('--- %s  ±%.2f rad ---' % (label, amp))
        print('   限位 [%.2f, %.2f]   起点 %.4f   峰值 %.4f (第 %d 条)   '
              '终点 %.4f'
              % (lo, hi, traj[0, idx], traj[peak, idx], peak + 1,
                 traj[-1, idx]))
        print('   峰值报文 q: %s' % np.array2string(traj[peak], precision=4))

        samples = {'t': [], 'q': []}
        stop = threading.Event()

        def sample(_stop=stop, _s=samples):
            try:
                sock = websocket.create_connection(ws_url, timeout=timeout)
            except Exception as exc:                  # noqa: BLE001
                print('   !! 采样连接失败: %s' % exc)
                return
            try:
                t0 = time.perf_counter()
                while not _stop.is_set():
                    try:
                        r = _request(sock, accid, 'request_get_joint_state',
                                     timeout)
                    except Exception:                 # noqa: BLE001
                        break
                    _s['t'].append(time.perf_counter() - t0)
                    _s['q'].append(np.asarray(r['q'], dtype=np.float64))
                    time.sleep(0.04)
            finally:
                sock.close()

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        time.sleep(0.3)
        stream_s, servo_stats = stream_trajectory(ws, accid, traj, hz)
        time.sleep(0.4)
        stop.set()
        th.join(timeout=5.0)

        print('   流: %d 条 / %.3f s = 实测 %.0f Hz；机器人回 %d 条，'
              '非 success %d 条'
              % (steps, stream_s, steps / max(stream_s, 1e-9),
                 servo_stats['replies'], servo_stats['failed']))
        if samples['q']:
            q = np.array(samples['q'])
            ref = q[0, idx]
            k = int(np.argmax(np.abs(q[:, idx] - ref)))
            span = max(samples['t'][-1], 1e-9)
            others = np.delete(np.arange(16), idx)
            print('   采样 %d 次 (%.0f Hz)' % (len(q), len(q) / span))
            print('   命令 ±%.4f  →  实测偏离 %.4f（第 %.2f s）'
                  % (amp, q[k, idx] - ref, samples['t'][k]))
            print('   结束时相对起点 %.4f' % (q[-1, idx] - ref))
            print('   其余 15 个关节最大偏离 %.4f'
                  % float(np.abs(q[:, others] - q[0, others]).max()))
        else:
            print('   !! 没采到关节数据')

        late = ws_drain(ws, 0.4)
        # Only response_* carries a result; the 1 Hz notify_robot_info push has
        # no `result` key and must not be counted as a failure.
        replies = [m for m in late
                   if str(m.get('title', '')).startswith('response_')]
        bad = sum(1 for m in replies
                  if m.get('data', {}).get('result') != 'success')
        print('   尾部回收 %d 条（response_* %d 条，其中非 success %d 条）'
              % (len(late), len(replies), bad))
        print()
        time.sleep(0.5)

    return 0


def run_selftest(mode: str, ws_url: str, accid: str, timeout: float = 6.0) -> int:
    """Probe the command channel with a command that asks for no motion.

    Every variant sends back the pose the robot already holds, so a correct
    implementation produces zero movement and we learn whether the channel and
    the message layout are accepted.  Keep a hand on the hardware emergency
    stop anyway: a hard stop is the only stop that works while moving.
    """
    import websocket

    ws = websocket.create_connection(ws_url, timeout=timeout)
    try:
        js = _request(ws, accid, 'request_get_joint_state', timeout)['q']
        js = np.asarray(js, dtype=np.float64)
        claw = _request(ws, accid, 'request_get_limx_2fclaw_state', timeout)
        print('当前关节(16) : %s' % np.array2string(js, precision=4))
        print('当前夹爪 hw  : left=%s right=%s'
              % (claw.get('left_opening'), claw.get('right_opening')))
        print()

        if mode == 'wiggle':
            return run_wiggle(ws, accid, js, ws_url, timeout)

        if mode == 'movej':
            data = {'joint': [float(v) for v in js[:14]], 'time': 2}
            print('>> request_movej  (14 维 = [L7, R7]，发当前位置)')
            print('   %s' % json.dumps(data))
        elif mode == 'servoj':
            n = 16
            data = {
                'filter_ratio': SERVOJ_FILTER_RATIO,
                'q': [float(v) for v in js[:16]],
                'v': [0.0] * n,
                'kp': SERVOJ_KP[:n],
                'kd': SERVOJ_KD[:n],
                'tau': [0.0] * n,
                'mode': [0] * n,
                'na': 0,
            }
            print('>> request_servoj (16 维 = [L7, R7, head2]，发当前位置)')
            print('   %s' % json.dumps(data))
        elif mode == 'servoj-stream':
            n = 16
            data = {
                'filter_ratio': SERVOJ_FILTER_RATIO,
                'q': [float(v) for v in js[:16]],
                'v': [0.0] * n,
                'kp': SERVOJ_KP[:n],
                'kd': SERVOJ_KD[:n],
                'tau': [0.0] * n,
                'mode': [0] * n,
                'na': 0,
            }
            steps = int(SERVOJ_STREAM_HZ * SERVOJ_STREAM_SECONDS)
            print('>> request_servoj 流  (16 维 = [L7, R7, head2]，恒发当前位置)')
            print('   %d Hz x %.1fs = %d 条；定节拍、不等响应（与部署同构）'
                  % (SERVOJ_STREAM_HZ, SERVOJ_STREAM_SECONDS, steps))
            print('   %s' % json.dumps(data))
            print()
            t0 = time.perf_counter()
            for step in range(1, steps + 1):
                ws_send_raw(ws, accid, 'request_servoj', data)
                target = t0 + step / SERVOJ_STREAM_HZ
                remaining = target - time.perf_counter()
                if remaining > 1e-3:
                    time.sleep(remaining - 1e-3)
                while time.perf_counter() < target:
                    pass
            stream_s = max(time.perf_counter() - t0, 1e-9)
            print('<< 流结束: %d 条 / %.3f s = 实测 %.0f Hz'
                  % (steps, stream_s, steps / stream_s))
            late = ws_drain(ws, 0.5)
            print('   尾部回收 %d 条消息:' % len(late))
            for msg in late[:5]:
                print('     %s' % json.dumps(msg, ensure_ascii=False)[:180])
        elif mode == 'gripper':
            right = float(claw.get('right_opening', 0.0))
            data = {'right_opening': right, 'right_speed': 100,
                    'right_force': 100}
            print('>> request_set_limx_2fclaw_cmd (发当前开口度，期望不动)')
            print('   %s' % json.dumps(data))
        else:
            raise SystemExit('未知 selftest 模式: %r' % mode)

        if mode != 'servoj-stream':
            title = {'movej': 'request_movej',
                     'servoj': 'request_servoj',
                     'gripper': 'request_set_limx_2fclaw_cmd'}[mode]
            print()
            resp = ws_request(ws, accid, title, data, timeout)
            print('<< 响应: %s' % json.dumps(resp, ensure_ascii=False))

        time.sleep(1.0)
        js2 = np.asarray(
            _request(ws, accid, 'request_get_joint_state', timeout)['q'],
            dtype=np.float64)
        moved = float(np.abs(js2 - js).max())
        print()
        print('指令后关节变化 max|Δ| = %.6f rad' % moved)
        print('判读: 应≈0（命令的就是当前位置）。若非 0，说明 16/14 维的顺序'
              '与假设不一致，先停下来重新核对。')
        return 0
    finally:
        ws.close()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shim', default='http://10.192.1.4:8770',
                    help='camera_shim base URL')
    ap.add_argument('--ws', default='ws://10.192.1.2:5000',
                    help='robot upper-layer WebSocket endpoint')
    ap.add_argument('--accid', default='DACH_TRON2A_215', help='robot SN')
    ap.add_argument('--zmq', default='tcp://127.0.0.1:5555',
                    help='inference server address')
    ap.add_argument('--unnorm-key', default='private')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--runs', type=int, default=3, help='number of iterations')
    ap.add_argument('--interval', type=float, default=1.0,
                    help='seconds between iterations')
    ap.add_argument('--save-dir', default=None,
                    help='optional directory to dump frames + actions as npz')
    ap.add_argument('--task', default='Press the red button',
                    help='训练过的任务文本，必须与数据集里的完全一致；'
                         '5 个任务是共用一个模型的唯一区分手段')
    ap.add_argument('--list-tasks', action='store_true',
                    help='列出训练过的全部任务并退出')
    ap.add_argument('--send-gripper', action='store_true',
                    help='允许下发右夹爪指令（拧旋钮任务需要）。'
                         '默认关闭：本程序默认绝不向机器人发送任何指令')
    ap.add_argument('--max-delta', type=float, default=DEFAULT_MAX_DELTA,
                    help='相邻两条 servoj 指令之间允许的最大增量(rad)，默认 %.2f'
                         % DEFAULT_MAX_DELTA)
    ap.add_argument('--max-reach', type=float, default=DEFAULT_MAX_REACH,
                    help='一条 chunk 相对起始实测位姿的最大包络(rad)，'
                         '默认 %.2f' % DEFAULT_MAX_REACH)
    ap.add_argument('--chunk-dt', type=float, default=DEFAULT_CHUNK_DT,
                    help='一个动作步的时间(s)，默认 1/30 = %.5f' % DEFAULT_CHUNK_DT)
    ap.add_argument('--execute', action='store_true',
                    help='⚠️ 真的把计划下发给机器人（默认关闭：只预览不发送）')
    ap.add_argument('--yes', action='store_true',
                    help='跳过 --execute / --goto-initial 的交互式确认')
    ap.add_argument('--goto-initial', action='store_true',
                    help='⚠️ 平滑移动到 --task 对应数据集片段的起始位姿'
                         '（需 --yes 或交互确认）')
    ap.add_argument('--goto-seconds', type=float, default=10.0,
                    help='--goto-initial 的运动时长(s)，默认 10')
    ap.add_argument('--selftest',
                    choices=['movej', 'servoj', 'servoj-stream', 'wiggle',
                             'gripper'],
                    default=None,
                    help='⚠️ 会真的向机器人发指令。movej/servoj/servoj-stream '
                         '发的是"保持当前位姿"（期望不动）；wiggle 会让单个关节'
                         '真的小幅动一下再回来。必须在机器人旁、手按硬急停时'
                         '运行')
    args = ap.parse_args()

    if args.selftest:
        print('=' * 74)
        print('  ⚠️  SELFTEST: 将向机器人发送一条指令（内容为当前位姿，期望不动）')
        print('=' * 74)
        print('  请确认：① 无人正在遥操   ② 手已放在本体硬急停上')
        print()
        return run_selftest(args.selftest, args.ws, args.accid)

    tasks = trained_tasks()
    if args.list_tasks:
        print('训练过的任务（%d 个）：' % len(tasks))
        for subset, text in sorted(tasks.items(), key=lambda kv: kv[1]):
            print('  %-42s  来自 %s' % (text, subset))
        return 0
    task = resolve_task(args.task, tasks)

    if args.goto_initial:
        import websocket

        subset = next((k for k, v in tasks.items() if v == task), None)
        if subset is None:
            print('!! 找不到 %r 对应的数据集片段' % task)
            return 2
        target14, spread, n_eps = initial_pose_for_task(subset)
        arm_names = ['abad_L', 'hip_L', 'yaw_L', 'knee_L', 'wy_L', 'wp_L',
                     'wr_L', 'abad_R', 'hip_R', 'yaw_R', 'knee_R', 'wy_R',
                     'wp_R', 'wr_R']

        ws_goto = websocket.create_connection(args.ws, timeout=6.0)
        try:
            cur16 = np.asarray(
                _request(ws_goto, args.accid, 'request_get_joint_state',
                         6.0)['q'], dtype=np.float64)[:16]
        finally:
            ws_goto.close()

        print('=' * 74)
        print('  ⚠️  goto-initial: 将把双臂移动到数据集的起始位姿')
        print('=' * 74)
        print('  任务     : %r' % task)
        print('  数据集   : %s  (%d 个片段的首帧均值)' % (subset, n_eps))
        print('  首帧离散度: 最大单关节标准差 %.4f rad（越小越可复现）' % spread)
        print('  运动时长 : %.1f s，raised-cosine（两端零速度）'
              % args.goto_seconds)
        print()
        print('  %-8s %10s %10s %10s  %-16s %s'
              % ('关节', '当前', '目标', 'Δ', '限位', '检查'))
        n_bad = 0
        for i, name in enumerate(arm_names):
            lo, hi = JOINT_LIMITS[i]
            ok = lo <= target14[i] <= hi
            n_bad += 0 if ok else 1
            print('  %-8s %10.4f %10.4f %+10.4f  [%6.2f,%6.2f]  %s'
                  % (name, cur16[i], target14[i], target14[i] - cur16[i],
                     lo, hi, 'OK' if ok else '!! 超限'))
        biggest = float(np.abs(target14 - cur16[:14]).max())
        print()
        print('  最大单关节位移: %.4f rad (%.1f 度)'
              % (biggest, np.degrees(biggest)))
        if n_bad:
            print('  !! %d 个关节的目标超出手册限位，拒绝执行。' % n_bad)
            return 2
        if not args.yes:
            try:
                answer = input('   确认请键入 yes 回车: ').strip().lower()
            except EOFError:
                answer = ''
            if answer != 'yes':
                print('   未确认，未发送任何指令。')
                return 3

        traj = ease_to_pose(cur16, target14, args.goto_seconds,
                            SERVOJ_STREAM_HZ)
        ws_goto = websocket.create_connection(args.ws, timeout=6.0)
        try:
            stream_s, servo_stats = stream_trajectory(ws_goto, args.accid,
                                                      traj, SERVOJ_STREAM_HZ)
            time.sleep(0.3)
            after = np.asarray(
                _request(ws_goto, args.accid, 'request_get_joint_state',
                         6.0)['q'], dtype=np.float64)[:16]
        finally:
            ws_goto.close()
        print()
        print('  [执行] %d 条 @ %d Hz，流 %.3f s (实测 %.0f Hz)；机器人回 %d 条，'
              '非 success %d 条'
              % (traj.shape[0], SERVOJ_STREAM_HZ, stream_s,
                 traj.shape[0] / max(stream_s, 1e-9),
                 servo_stats['replies'], servo_stats['failed']))
        print('  实测相对目标 max|Δ|: %.4f rad'
              % float(np.abs(after[:14] - target14).max()))
        print('  执行后 /joint_states(16): %s'
              % np.array2string(after, precision=4))
        return 0

    print('=' * 74)
    if args.execute:
        print('  TRON2 客户端  ---  ⚠️⚠️ 执行模式：将真的驱动机器人')
    elif args.send_gripper:
        print('  TRON2 客户端  ---  ⚠️ 已启用夹爪下发（会产生动作）')
    else:
        print('  TRON2 只读回环客户端  ---  不会向机器人发送任何指令')
    print('=' * 74)
    print('  cameras : %s' % args.shim)
    print('  joints  : %s' % args.ws)
    print('  server  : %s' % args.zmq)
    print('  task    : %r' % task)
    print('  夹爪    : 输入读取真值 (hw 0-100 / %.0f)；下发 %s'
          % (GRIPPER_SCALE,
             '已启用（右夹爪 act[17] × 100）' if args.send_gripper
             else '已禁用'))

    st = shim_status(args.shim)
    if 'error' in st:
        print()
        print('!! camera shim 不可用: %s' % st['error'])
        return 2
    for name, info in st.get('cameras', {}).items():
        print('  cam %-6s ok=%-5s frames=%s age=%sms'
              % (name, info.get('ok'), info.get('frames'), info.get('age_ms')))
    print()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    ws_servo = None
    if args.execute:
        print('!! 执行模式：将以 servoj 向机器人下发策略输出。')
        print('!! 手臂行为由策略决定；硬急停是唯一可靠的停止手段。')
        print('!! 每个 chunk 50 步 x %.4f s = %.2f s；本次 %d 个 chunk'
              ' = 约 %.1f s 运动。'
              % (args.chunk_dt, 50 * args.chunk_dt, args.runs,
                 args.runs * 50 * args.chunk_dt))
        if not args.yes:
            try:
                answer = input('   确认请键入 yes 回车: ').strip().lower()
            except EOFError:
                answer = ''
            if answer != 'yes':
                print('   未确认，退出。')
                return 3
        import websocket
        ws_servo = websocket.create_connection(args.ws, timeout=6.0)
        print('   servoj 连接已建立。')
    print()

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    for i in range(1, args.runs + 1):
        t0 = time.perf_counter()
        frames = fetch_frames(args.shim)
        js16, grip2 = read_robot_state(args.ws, args.accid)
        qpos, states = build_observation(js16, grip2)
        t_obs = time.perf_counter() - t0

        obs = dict(frames)
        obs['qpos'] = qpos
        obs['states'] = states
        obs['task_description'] = task

        t1 = time.perf_counter()
        actions, infer_s = predict(obs, args.zmq, args.unnorm_key, args.seed)
        t_round = time.perf_counter() - t1

        print('--- 第 %d 次 ---' % i)
        print('  观测耗时 %.0f ms  推理往返 %.0f ms (服务端 %.1f ms)'
              % (t_obs * 1000, t_round * 1000, infer_s * 1000))
        print('  图像        : %s' % {
            k: tuple(v.shape) + (int(v.mean()),) for k, v in frames.items()})
        print('  /joint_states(16): %s' % np.array2string(js16, precision=4))
        print('  夹爪 hw/标定后   : [%.0f, %.0f] -> [%.4f, %.4f]'
              % (grip2[0] * GRIPPER_SCALE, grip2[1] * GRIPPER_SCALE,
                 grip2[0], grip2[1]))
        print('  qpos(16) -> 服务端: %s' % np.array2string(qpos, precision=4))
        print('  states(18) 服务端 : %s' % np.array2string(states, precision=4))
        print('  动作 shape=%s denormalized' % (actions.shape,))
        print('  第一步(18)  : %s' % np.array2string(actions[0], precision=4))
        print('  末步(18)    : %s' % np.array2string(actions[-1], precision=4))
        print('  右臂相对当前位姿 max|Δ|: %.4f'
              % float(np.abs(actions[0, 7:14] - states[7:14]).max()))
        print('  左臂相对当前位姿 max|Δ|: %.4f'
              % float(np.abs(actions[0, 0:7] - states[0:7]).max()))
        print('  左夹爪 idx16 min/max: %.4f / %.4f  (§7.2 期望恒为 0)'
              % (actions[:, 16].min(), actions[:, 16].max()))
        print('  右夹爪 idx17 min/max: %.4f / %.4f'
              % (actions[:, 17].min(), actions[:, 17].max()))
        print('  头部 idx14-15       : %s' % np.array2string(actions[0, 14:16]))
        print('  finite              : %s' % bool(np.isfinite(actions).all()))

        # Safety layer.  Without --execute this is a preview: the plan below is
        # what *would* be sent, and nothing reaches the robot.
        plan, info = plan_commands(actions, states, max_delta=args.max_delta,
                                   max_reach=args.max_reach)
        cur16 = np.concatenate([states[0:7], states[7:14], states[14:16]])
        raw = np.concatenate([actions[:, 0:7], actions[:, 7:14],
                              actions[:, 14:16]], axis=1)
        tag = '执行' if args.execute else '预览不发送'
        print()
        print('  [安全层·%s] servoj 计划  (--max-delta %.2f / --max-reach %.2f)'
              % (tag, args.max_delta, args.max_reach))
        print('     左臂 7 维 : 已锁定为实测位姿（模型输出被丢弃）')
        print('     步间限幅  : %d 个动作值被 --max-delta 限住'
              % info['rate_clamped'])
        print('     包络限幅  : %d 个动作值被 --max-reach 限住'
              % info['reach_clamped'])
        print('     绝对限位  : %d 个动作值被手册限位限住' % info['abs_clamped'])
        print('     相对当前位姿 max|Δ|: 原始 %.4f -> 钳制后 %.4f'
              % (float(np.abs(raw - cur16).max()),
                 float(np.abs(plan - cur16).max())))
        print('     计划末端相对起点 max|Δ|: %.4f rad'
              % float(np.abs(plan[-1] - cur16).max()))
        print('     servoj q(16) 第一步: %s'
              % np.array2string(plan[0], precision=4))
        print('  [安全层·%s] 夹爪 (模型 0-1 × %.0f = 硬件 0-100):'
              % (tag, GRIPPER_SCALE))
        print('     request_set_limx_2fclaw_cmd right_opening = %.0f  '
              '(模型输出 idx17=%.4f)'
              % (gripper_command(actions[0, 17]), actions[0, 17]))
        print('     左夹爪: 保持不动（5 个子集里恒为 0，退化维度）'
              if LEFT_GRIPPER_DEGENERATE else '     左夹爪: 下发')
        print('     --send-gripper: %s'
              % ('已启用' if args.send_gripper else '未启用（仅预览）'))
        print()

        if args.execute and ws_servo is not None:
            ref = interpolate_plan(plan, args.chunk_dt, SERVOJ_STREAM_HZ)
            print('  [执行] servoj %d 个 waypoint -> %d 条 @ %d Hz (%.2f s)'
                  % (plan.shape[0], ref.shape[0], SERVOJ_STREAM_HZ,
                     ref.shape[0] / SERVOJ_STREAM_HZ))
            stream_s, servo_stats = stream_trajectory(ws_servo, args.accid,
                                                      ref, SERVOJ_STREAM_HZ)
            time.sleep(0.2)
            after = read_robot_state(args.ws, args.accid)[0]
            print('     流 %.3f s (实测 %.0f Hz)；机器人回 %d 条，非 success %d 条'
                  % (stream_s, ref.shape[0] / max(stream_s, 1e-9),
                     servo_stats['replies'], servo_stats['failed']))
            print('     执行后 /joint_states(16): %s'
                  % np.array2string(after, precision=4))
            print('     实测相对计划末端 max|Δ|: %.4f rad'
                  % float(np.abs(after - plan[-1]).max()))
            print()
            if args.save_dir:
                np.savez_compressed(
                    os.path.join(args.save_dir, 'chunk_%03d.npz' % i),
                    plan=plan, qpos=qpos, states=states, actions=actions,
                    **frames)

        if args.save_dir:
            path = os.path.join(args.save_dir, 'obs_%03d.npz' % i)
            np.savez_compressed(path, qpos=qpos, states=states,
                                actions=actions, **frames)
            print('  已保存 %s' % path)
            print()

        if i < args.runs:
            time.sleep(args.interval)

    if ws_servo is not None:
        ws_servo.close()
    if args.execute:
        print('完成。执行模式已停止；机器人保持最后一条 servoj 指令的位置。')
    else:
        print('完成。提醒：本程序没有执行任何动作，机器人全程未收到指令。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
