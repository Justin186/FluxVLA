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

# Same order, so a deviation measured in a joint trace can be named directly.
JOINT_NAMES = ['abad_L', 'hip_L', 'yaw_L', 'knee_L', 'wy_L', 'wp_L', 'wr_L',
               'abad_R', 'hip_R', 'yaw_R', 'knee_R', 'wy_R', 'wp_R', 'wr_R',
               'head_pitch', 'head_yaw']

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
#
# It has to clear the policy's own output, otherwise the rail stops being a rail
# and becomes part of the controller.  Measured on the TRON2 cabinet policy: it
# asks for 0.83-0.92 rad within a chunk, so the earlier 0.80 default clipped
# every reach and the arm fell short of the button.  The old recordings are also
# a sanity check on the scale -- their whole press moves at most 0.75 rad.  Set
# this well above the observed spread, not at its edge.
DEFAULT_MAX_REACH = 1.20          # rad, ~69 deg per chunk
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


def send_gripper(ws, accid: str, right01: float):
    """Command the right gripper.  Model output is 0-1, the robot wants 0-100.

    Speed and force follow ``Tron2Operator._send_gripper``.  The left gripper is
    never commanded: it is constant 0 across all five training subsets, so the
    policy has no signal for it and the recordings never move it.
    """
    ws_send_raw(ws, accid, 'request_set_limx_2fclaw_cmd', {
        'right_opening': float(np.clip(right01, 0.0, 1.0) * GRIPPER_SCALE),
        'right_speed': 100,
        'right_force': 100,
    })


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


# Cap on how fast the fastest joint may travel during a goto.  The training
# start pose is far from wherever a restart leaves the robot -- on this machine
# yaw_R alone has 1.76 rad (101 deg) to travel -- so the duration is derived
# from the distance rather than fixed, and the operator gets a number that
# means something when deciding whether the path is clear.
DEFAULT_GOTO_VEL = 0.15           # rad/s, roughly 8.6 deg/s


def goto_seconds_for(target14: np.ndarray, cur16: np.ndarray,
                     max_vel, fallback: float) -> float:
    """Duration for ``ease_to_pose`` so the peak joint velocity is ``max_vel``.

    The raised-cosine gain's peak slope is ``span * pi / (2 * T)``, so
    ``T = span * pi / (2 * max_vel)`` holds the fastest joint at the cap.
    A ``max_vel`` of 0 or None keeps ``fallback`` seconds instead.
    """
    if max_vel is None or max_vel <= 0:
        return float(fallback)
    span = float(np.abs(np.asarray(target14)
                        - np.asarray(cur16)[:len(target14)]).max())
    return max(span * np.pi / (2.0 * float(max_vel)), 0.5)


# Order in which the arm joints travel to the training start pose, in servoj
# order.  Interpolating all 14 at once turns the shoulder yaw and bends the
# elbow in the same breath, so the forearm sweeps a wide arc and on this robot
# puts it into the table in front.  Doing it in groups keeps the hands clear:
# orient the wrists first, then bring the shoulders and elbows round, and only
# then swing the shoulder yaw through its 85 degrees.
GOTO_STAGES = [
    # Elbow first, upper arm still.  Measured on the robot: folding from the
    # home pose sends the forearm *outward*, clear of the table and of the
    # legs, and it is the shoulder yaw roll that decides which way the fold
    # goes.  A 180 degree pre-roll flipped it inward and drove the forearm into
    # the operator's legs, so the roll is deliberately left out and yaw is
    # turned last instead.  Add "2=~-3.1416,9=~3.1416" as a first stage only
    # if you want that flip back.
    ('小臂往外举 90 度', [(3, None), (10, None)]),        # elbow
    ('手臂离开双腿、前后到位', [(1, None), (8, None), (0, None), (7, None)]),
    ('手腕到位',       [(4, None), (5, None), (6, None),
                        (11, None), (12, None), (13, None)]),
    ('大臂转回 90 度', [(2, None), (9, None)]),           # shoulder yaw, last
]


def parse_stage_spec(spec: str):
    """Parse a stage spec into ``[(label, [(joint, value_or_None), ...]), ...]``.

    Stages are separated by ``;`` and joints within a stage by ``,``.  A token
    is either a joint index (move it to its final target now) or
    ``index=value`` (move it to that value now -- a temporary dodge -- and let
    a later stage bring it to the target).  Example::

        0=-1.2,7=-1.2; 3,10; 2,9; 0,7
    """
    out = []
    for n, chunk in enumerate(str(spec).split(';')):
        items = []
        for token in chunk.replace(' ', '').split(','):
            if not token:
                continue
            if '=' in token:
                key, val = token.split('=', 1)
                if val.startswith('~'):
                    # "~offset" is measured from where the joint is right now,
                    # so a dodge written as "spin 180 degrees" stays a true 180
                    # whatever pose the move happens to start from.
                    items.append((int(key), ('rel', float(val[1:]))))
                else:
                    items.append((int(key), float(val)))
            else:
                items.append((int(token), None))
        if items:
            out.append(('阶段%d' % (n + 1), items))
    return out


def stage_goto(cur16: np.ndarray, target14: np.ndarray, stages, max_vel: float,
               hz: int, frac: float = 1.0):
    """One continuous reference that moves the arm joints a stage at a time.

    Each stage eases only its own joints toward the target and holds the rest,
    so the operator can order the motion to clear whatever is in the way.  The
    stages are concatenated into a *single* stream: pausing between them would
    drop servo control and re-engage it, which is exactly the hand-over that
    bangs when the brakes let go.

    Returns ``(traj, plan)`` where ``plan`` is ``[(label, indices, seconds)]``
    for reporting.
    """
    target14 = np.asarray(target14, dtype=np.float64)
    cur = np.asarray(cur16, dtype=np.float64)[:16].copy()
    rows = []
    plan = []
    for label, items in stages:
        pairs = []
        for i, v in items:
            if not (0 <= i < 14):
                continue
            if v is None:
                dst = float(target14[i])
            elif isinstance(v, tuple):        # ('rel', offset)
                dst = float(cur[i]) + v[1]
            else:
                dst = float(v)
            # frac < 1 walks only part of the way -- enough to see which way a
            # joint is heading before committing to the whole move.
            pairs.append((i, float(cur[i]) + frac * (dst - float(cur[i]))))
        if not pairs:
            continue
        span = max(abs(v - cur[i]) for i, v in pairs)
        if span < 1e-9:
            plan.append((label, pairs, 0.0))
            continue
        seconds = max(span * np.pi / (2.0 * max_vel), 0.5)
        steps = max(int(round(seconds * hz)), 2)
        gain = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, steps))
        block = np.tile(cur, (steps, 1))
        for i, v in pairs:
            block[:, i] = cur[i] + (v - cur[i]) * gain
        rows.append(block)
        for i, v in pairs:
            cur[i] = v
        plan.append((label, pairs, seconds))
    if not rows:
        return np.asarray(cur, dtype=np.float64)[None, :], plan
    return np.vstack(rows), plan


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


def stream_trajectory(ws, accid: str, traj: np.ndarray, hz: int,
                      gripper=None, sub_steps=None):
    """Push a [T, 16] trajectory as a servoj stream.

    Returns ``(elapsed_seconds, stats)`` where ``stats`` counts the robot's
    replies and how many of them were not ``success``.

    ``gripper`` is an optional per-waypoint right-gripper trajectory (0-1) and
    ``sub_steps`` is how many servo messages each waypoint is expanded into.
    When both are given the gripper is commanded once per waypoint, which is
    what ``Tron2Operator._run_trajectory_servoj`` does -- the selector tasks
    need it, the button tasks leave the gripper at 0 throughout.

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
    state = {'stop': False, 'replies': 0, 'failed': 0, 'faults': []}

    def drainer():
        ws.settimeout(0.3)
        while not state['stop']:
            try:
                msg = json.loads(ws.recv())
            except Exception:            # noqa: BLE001
                continue
            title = str(msg.get('title', ''))
            data = msg.get('data') or {}
            if title.startswith('response_'):
                state['replies'] += 1
                res = data.get('result')
                if res != 'success':
                    state['failed'] += 1
                    # Keep the reasons, not just the count: a rejected command
                    # is the robot refusing to move, and on this machine it
                    # means the motor faulted (measured: fail_motor during a
                    # DC link under-voltage while moving fast under load).
                    if len(state['faults']) < 6:
                        state['faults'].append('%s (response #%d)'
                                               % (res, state['replies']))
            elif title == 'notify_servoJ':
                # SDK guide 3.6.4.3: servoj has no reply, so this push is the
                # only signal that a command was rejected -- fail_motor means
                # the motor itself faulted.  Missing it would let a whole
                # trajectory run while the arm refuses to move and nothing
                # anywhere says so.
                state['faults'].append(str(data.get('result')))

    reader = threading.Thread(target=drainer, daemon=True)
    reader.start()

    t0 = time.perf_counter()
    last_seg = -1
    for i in range(steps):
        if gripper is not None and sub_steps:
            seg = i // sub_steps
            if seg != last_seg:
                send_gripper(ws, accid,
                             float(gripper[min(seg, len(gripper) - 1)]))
                last_seg = seg
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


def start_joint_watch(ws_url: str, accid: str, timeout: float = 6.0,
                      period: float = 0.006):
    """Poll q/dq/tau on a dedicated connection while a stream runs.

    Taking over servo control is a millisecond-scale event, so the hand-over is
    invisible to anything that is not already sampling.  That is how a yaw
    joint parked against its mechanical stop can thump at start-up while every
    recorded trace looks flat.  tau is kept because a joint driven into a stop
    shows a sustained torque spike even when the position hardly moves.

    Returns ``(stop_and_join, samples)``.
    """
    import threading

    import websocket

    samples = {'t': [], 'q': [], 'dq': [], 'tau': []}
    stop = threading.Event()

    def run():
        try:
            sock = websocket.create_connection(ws_url, timeout=timeout)
        except Exception as exc:                      # noqa: BLE001
            print('  [warn] 监控连接失败: %s' % exc)
            return
        try:
            t0 = time.perf_counter()
            while not stop.is_set():
                try:
                    d = _request(sock, accid, 'request_get_joint_state',
                                 timeout)
                except Exception:                     # noqa: BLE001
                    break
                samples['t'].append(time.perf_counter() - t0)
                samples['q'].append(np.asarray(d['q'], dtype=np.float64))
                samples['dq'].append(np.asarray(d.get('dq', [np.nan] * 16),
                                                dtype=np.float64))
                samples['tau'].append(np.asarray(d.get('tau', [np.nan] * 16),
                                                 dtype=np.float64))
                time.sleep(period)
        finally:
            sock.close()

    th = threading.Thread(target=run, daemon=True)
    th.start()

    def stop_and_join():
        stop.set()
        th.join(timeout=3.0)

    return stop_and_join, samples


def report_joint_watch(samples: dict, t_takeover, window: float = 0.4):
    """Summarise a watch around the moment servo control was taken over."""
    if not samples['t'] or t_takeover is None:
        print('  [监控] 没采到数据')
        return
    T = np.asarray(samples['t'])
    Q = np.asarray(samples['q'])
    TAU = np.asarray(samples['tau'])
    ref = np.median(Q[:3], axis=0)
    k0 = int(np.searchsorted(T, t_takeover - window))
    k1 = int(np.searchsorted(T, t_takeover + window))
    print('  [监控] %d 次采样 / %.2f s (%.0f Hz)；接管瞬间 ±%.0f ms'
          % (len(T), T[-1], len(T) / max(T[-1], 1e-9), window * 1000))
    if k1 <= k0:
        print('         窗口内无采样点')
        return
    dev = np.abs(Q[k0:k1] - ref).max(axis=0)
    j = int(np.argmax(dev))
    print('         接管窗口内位移最大 : %s %.4f rad (%.2f 度)'
          % (JOINT_NAMES[j], dev[j], np.degrees(dev[j])))
    moved = [(JOINT_NAMES[i], float(dev[i])) for i in range(16)
             if dev[i] > 0.005 and i != j]
    if moved:
        # Several joints moving at once, including ones nobody commanded, is
        # the signature of a whole-robot transient (a drive hand-over) rather
        # than of one joint misbehaving.
        print('         窗口内还动过的关节(%d): %s'
              % (len(moved), ', '.join('%s %.3f' % t for t in moved)))
    seg = np.abs(TAU[k0:k1])
    amax = np.unravel_index(int(np.nanargmax(seg)), seg.shape)
    print('         接管窗口内 |tau| 最大: %.2f  (关节 %s)'
          % (float(seg[amax]), JOINT_NAMES[amax[1]]))
    print('         全程 |tau| 最大     : %.2f'
          % float(np.nanmax(np.abs(TAU))))

    # Print the shapes, not just the peaks: a real excursion rises and falls
    # across consecutive samples, a bad frame is one point that jumps to an
    # unrelated value and straight back.  The two need opposite conclusions.
    tj = int(amax[1])
    for label, jj in (('位移最大关节', j), ('力矩最大关节', tj)):
        if not (k0 < k1):
            continue
        print('         [%s] %s 的轨迹:' % (label, JOINT_NAMES[jj]))
        step = max(1, (k1 - k0) // 12)
        for k in range(k0, k1, step):
            print('           t%+7.3f   %-9s %8.4f   tau %8.2f'
                  % (T[k] - t_takeover, JOINT_NAMES[jj], Q[k, jj],
                     TAU[k, jj]))


def servo_summary(stats: dict) -> str:
    """Describe what came back on a servo socket, faults included.

    ``response_servoj`` says ``success`` even for commands the robot then
    ignores, so a clean reply count is not evidence of anything; the only
    server-side failure signal is the ``notify_servoJ`` push.
    """
    out = '机器人回 %d 条，非 success %d 条' % (stats['replies'],
                                                stats['failed'])
    faults = stats.get('faults') or []
    if faults:
        out += '\n      !! 拒绝原因: %s' % '; '.join(faults[:4])
    return out


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


# How long the reference takes to ease from the freshly measured pose into the
# first planned waypoint.  ServoJ holds with tau ~ kp * (q_ref - q_actual) and
# kp reaches 420, so starting a stream from a reference that does not match the
# arm asks the motor for the whole correction in a single control cycle.  With
# the arms hanging limp during start-up the gap is at its largest, which is why
# the thump happens then.
ENGAGE_SECONDS = 0.4


def engage_ramp(target: np.ndarray, fresh: np.ndarray, hz: int,
                seconds: float = ENGAGE_SECONDS) -> np.ndarray:
    """Ease from ``fresh`` (just measured) into ``target`` (first waypoint).

    Returns [steps, 16].  The raised-cosine gain has zero slope at both ends,
    so neither entering the stream nor handing over to the plan introduces a
    velocity step.
    """
    steps = max(int(round(seconds * hz)), 1)
    u = np.linspace(0.0, 1.0, steps)        # 0 and 1 included, so the first row
    gain = 0.5 - 0.5 * np.cos(np.pi * u)    # is exactly `fresh` and the last is
    return fresh + (target - fresh) * gain[:, None]     # exactly `target`


def run_wiggle(ws, accid: str, js, ws_url: str, timeout: float = 6.0,
               joints=None) -> int:
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

    for idx, amp, label in (joints or WIGGLE_JOINTS):
        lo, hi = JOINT_LIMITS[idx]
        traj = out_and_back(base, idx, amp, steps)
        # Check what the trajectory actually visits, not the symmetric +-amp
        # band: a one-sided wiggle is legal even when the arm is already parked
        # against one end of the range (which is where a restart leaves yaw).
        tmin = float(traj[:, idx].min())
        tmax = float(traj[:, idx].max())
        # Never narrow the bounds below where the robot already is: a restart
        # parks yaw against the stop, reading 1.4822 against a documented 1.48,
        # and a joint sitting there must still be allowed to move inward.
        lo = min(lo, float(base[idx]))
        hi = max(hi, float(base[idx]))
        if not (lo <= tmin and tmax <= hi):
            print('--- %s 跳过: 轨迹 [%.3f, %.3f] 超出限位 [%.2f, %.2f]'
                  % (label, tmin, tmax, lo, hi))
            print()
            continue

        peak = int(np.argmax(np.abs(traj[:, idx] - traj[0, idx])))
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

        print('   流: %d 条 / %.3f s = 实测 %.0f Hz；%s'
              % (steps, stream_s, steps / max(stream_s, 1e-9),
                 servo_summary(servo_stats)))
        if samples['q']:
            q = np.array(samples['q'])
            # The first frame off a freshly opened connection can be stale, and
            # using it as the reference invents a deviation that never happened
            # (seen on this robot: a phantom 1.10 rad on yaw_L while yaw_L was
            # commanded constant).  The median of the opening frames is safe.
            ref_row = np.median(q[:3], axis=0) if len(q) > 2 else q[0]
            ref = ref_row[idx]
            k = int(np.argmax(np.abs(q[:, idx] - ref)))
            span = max(samples['t'][-1], 1e-9)
            others = np.delete(np.arange(16), idx)
            print('   采样 %d 次 (%.0f Hz)' % (len(q), len(q) / span))
            print('   命令 ±%.4f  →  实测偏离 %.4f（第 %.2f s）'
                  % (amp, q[k, idx] - ref, samples['t'][k]))
            print('   结束时相对起点 %.4f' % (q[-1, idx] - ref))
            dev = np.abs(q[:, others] - ref_row[others]).max(axis=0)
            k2 = int(np.argmax(dev))
            print('   其余 15 个关节最大偏离 %.4f   关节=%s'
                  % (float(dev.max()), JOINT_NAMES[others[k2]]))
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


def run_selftest(mode: str, ws_url: str, accid: str, timeout: float = 6.0,
                 wiggle_joints=None) -> int:
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
            return run_wiggle(ws, accid, js, ws_url, timeout, wiggle_joints)

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
            print('   %d Hz x %.1fs = %d 条；定节拍、边发边收（与部署同构）'
                  % (SERVOJ_STREAM_HZ, SERVOJ_STREAM_SECONDS, steps))
            print('   %s' % json.dumps(data))
            print()
            traj = np.tile(np.asarray(js, dtype=np.float64)[:16], (steps, 1))
            stream_s, servo_stats = stream_trajectory(
                ws, accid, traj, SERVOJ_STREAM_HZ)
            print('<< 流结束: %d 条 / %.3f s = 实测 %.0f Hz；%s'
                  % (steps, stream_s, steps / max(stream_s, 1e-9),
                     servo_summary(servo_stats)))
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
    ap.add_argument('--engage-time', type=float, default=ENGAGE_SECONDS,
                    help='从"发指令前实测位姿"平滑过渡到计划首步的时间(s)，'
                         '默认 %.1f。用于消除进入伺服时的力矩台阶'
                         % ENGAGE_SECONDS)
    ap.add_argument('--execute', action='store_true',
                    help='⚠️ 真的把计划下发给机器人（默认关闭：只预览不发送）')
    ap.add_argument('--yes', action='store_true',
                    help='跳过 --execute / --goto-initial 的交互式确认')
    ap.add_argument('--goto-initial', action='store_true',
                    help='⚠️ 平滑移动到 --task 对应数据集片段的起始位姿'
                         '（需 --yes 或交互确认）')
    ap.add_argument('--goto-seconds', type=float, default=10.0,
                    help='--goto-initial 的运动时长(s)；仅当 --goto-vel 为 0 '
                         '时用作回退值，默认 10')
    ap.add_argument('--goto-vel', type=float, default=DEFAULT_GOTO_VEL,
                    help='--goto-initial 的峰值关节速度上限(rad/s)，默认 '
                         '%.2f；运动时长由距离推导。设为 0 则改用 '
                         '--goto-seconds' % DEFAULT_GOTO_VEL)
    ap.add_argument('--goto-stages', type=str, default=None,
                    help='--goto-initial 分阶段运动的关节分组，用分号分隔各'
                         '阶段，阶段内用逗号列关节索引，如 "4,5,6;0,1,3;2,9"。'
                         '默认按 GOTO_STAGES（先腕、再肩肘、最后肩部 yaw）')
    ap.add_argument('--goto-single', action='store_true',
                    help='--goto-initial 退回到"所有关节一次插值"（会扫出大弧，'
                         '可能碰到前面的桌子，仅在你确认路径空旷时用）')
    ap.add_argument('--goto-only', type=str, default=None,
                    help='--goto-initial 只跑指定的阶段（逗号分隔，按 1 起编号，'
                         '如 "1" 或 "1,2"）。用来一步一步验证路径')
    ap.add_argument('--goto-frac', type=float, default=1.0,
                    help='--goto-initial 只走该阶段行程的这个比例，默认 1.0。'
                         '如 0.1 = 只动十分之一，用来先确认方向')
    ap.add_argument('--wiggle-joint', type=str, default=None,
                    help='--selftest wiggle 时只摆动这些关节索引（逗号分隔，'
                         '如 "0,7"）；默认摆 WIGGLE_JOINTS 里的两个。用来实测'
                         '某个关节的正负方向')
    ap.add_argument('--wiggle-amp', type=float, default=0.15,
                    help='--selftest wiggle 的摆动幅度(rad)，默认 0.15')
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
        joints = None
        if args.wiggle_joint:
            joints = []
            for token in str(args.wiggle_joint).replace(' ', '').split(','):
                if not token:
                    continue
                # "idx=amp" lets each joint pick its own sign, which mirrored
                # joints need: yaw sits at +1.48 on the left and -1.48 on the
                # right, so "inward" is negative for one and positive for the
                # other.
                if '=' in token:
                    key, val = token.split('=', 1)
                    idx, amp = int(key), float(val)
                else:
                    idx, amp = int(token), args.wiggle_amp
                joints.append((idx, amp, 'joint idx%d 方向实测（%+.2f rad）'
                               % (idx, amp)))
        return run_selftest(args.selftest, args.ws, args.accid,
                            wiggle_joints=joints)

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
        arm_names = JOINT_NAMES[:14]

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

        if args.goto_single:
            stages = [('一次插值（全部关节）',
                       [(i, None) for i in range(14)])]
        else:
            stages = (parse_stage_spec(args.goto_stages) if args.goto_stages
                      else list(GOTO_STAGES))
        only = None
        if args.goto_only:
            only = [int(t) for t in str(args.goto_only).replace(' ', '')
                    .split(',') if t]
            bad = [n for n in only if not 1 <= n <= len(stages)]
            if bad:
                print('  !! --goto-only 指定的阶段不存在: %s（共 %d 个阶段）'
                      % (bad, len(stages)))
                return 2
            stages = [s for n, s in enumerate(stages, 1) if n in only]
        vel = args.goto_vel
        if not vel or vel <= 0:
            # --goto-vel 0 means "read --goto-seconds as the duration".
            secs = goto_seconds_for(target14, cur16, None, args.goto_seconds)
            span0 = float(np.abs(target14 - cur16[:14]).max())
            vel = max(span0 * np.pi / (2.0 * secs), 1e-3)
        _, stage_plan = stage_goto(cur16, target14, stages, vel,
                                   SERVOJ_STREAM_HZ, args.goto_frac)
        total_s = sum(sec for _, _, sec in stage_plan)
        covered = {i for _, pairs, _ in stage_plan for i, _ in pairs}
        missed = [arm_names[i] for i in range(14)
                  if abs(target14[i] - cur16[i]) > 1e-9 and i not in covered]
        print('  运动时长 : %.1f s，分 %d 阶段（每阶段 raised-cosine 两端零速度）'
              % (total_s, len([1 for _, _, sec in stage_plan if sec > 0])))
        print('  峰值关节速度上限: %.3f rad/s = %.1f 度/s'
              % (vel, np.degrees(vel)))
        if only is not None:
            print('  ⚠️  本次只跑第 %s 阶段，其余阶段这轮不动'
                  % ','.join(str(n) for n in only))
        if args.goto_frac < 1.0:
            print('  ⚠️  本次只走 %.0f%% 的行程（先确认方向，不走到目标）'
                  % (args.goto_frac * 100))
        print()
        print('  阶段顺序（"(临时)" = 中途躲避位置，不是最终目标）：')
        for label, pairs, sec in stage_plan:
            print('    %s   %.1f s' % (label, sec))
            for i, v in pairs:
                tag = '' if abs(v - target14[i]) < 1e-9 else '   (临时)'
                print('        %-9s %8.3f -> %8.3f%s'
                      % (arm_names[i], cur16[i], v, tag))
        if missed and only is None:
            print()
            print('  !! 以下关节需要移动但未被任何阶段覆盖，将保持不动：')
            print('     %s' % ' '.join(missed))
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

        # The table above was built from a pose read before the confirmation
        # prompt, so it can be seconds stale by the time we stream.  ServoJ
        # holds with tau ~ kp * (q_ref - q_actual), so a stale reference leaves
        # a gap that the first message asks the motor to close in one cycle.
        # Re-read here and rebuild the trajectory from what the robot says now.
        ws_goto = websocket.create_connection(args.ws, timeout=6.0)
        try:
            fresh = np.asarray(
                _request(ws_goto, args.accid, 'request_get_joint_state',
                         6.0)['q'], dtype=np.float64)[:16]
            drift = float(np.abs(fresh[:14] - cur16[:14]).max())
            if drift > 1e-4:
                print('  确认期间手臂移动了 %.4f rad，已按新位姿重建轨迹'
                      % drift)
            traj, _ = stage_goto(fresh, target14, stages, vel,
                                 SERVOJ_STREAM_HZ, args.goto_frac)
            stop_watch, watch = start_joint_watch(args.ws, args.accid)
            time.sleep(0.5)                 # baseline before control is taken
            t_takeover = watch['t'][-1] if watch['t'] else None
            stream_s, servo_stats = stream_trajectory(ws_goto, args.accid,
                                                      traj, SERVOJ_STREAM_HZ)
            time.sleep(0.4)
            stop_watch()
            after = np.asarray(
                _request(ws_goto, args.accid, 'request_get_joint_state',
                         6.0)['q'], dtype=np.float64)[:16]
        finally:
            ws_goto.close()
        print()
        print('  [执行] %d 条 @ %d Hz，流 %.3f s (实测 %.0f Hz)；%s'
              % (traj.shape[0], SERVOJ_STREAM_HZ, stream_s,
                 traj.shape[0] / max(stream_s, 1e-9),
                 servo_summary(servo_stats)))
        # Compare against where THIS run was aiming, not the overall target:
        # with --goto-only or --goto-frac the run deliberately stops short, and
        # measuring against the final pose would look like a huge error.
        print('  实测相对【本轮终点】 max|Δ|: %.4f rad'
              % float(np.abs(after[:14] - traj[-1][:14]).max()))
        report_joint_watch(watch, t_takeover)
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
            # Anchor the stream to the pose the robot is in *right now*, not the
            # pose the observation was taken at.  Inference sits between the two
            # (2.6 s on the first chunk) and the arms hang limp during start-up,
            # so they sag in that gap; starting from plan[0] would ask the motor
            # for the whole correction in one cycle.
            fresh = np.asarray(read_robot_state(args.ws, args.accid)[0][:16],
                               dtype=np.float64)
            gap = float(np.abs(fresh - cur16).max())
            ref = interpolate_plan(plan, args.chunk_dt, SERVOJ_STREAM_HZ)
            head = engage_ramp(ref[0], fresh, SERVOJ_STREAM_HZ,
                               args.engage_time)
            sub_steps = max(int(round(args.chunk_dt * SERVOJ_STREAM_HZ)), 1)
            grip = None
            if args.send_gripper:
                grip = np.clip(actions[:, 17], 0.0, 1.0)
                # Hold the gripper through the engage ramp so its segment index
                # stays aligned with the action steps.
                head_segs = int(round(head.shape[0] / sub_steps))
                grip = np.concatenate([np.full(head_segs, grip[0]), grip])
                print('  [执行] 夹爪：右夹爪 %d 个 waypoint，目标 %.2f~%.2f '
                      '(硬件 %.0f~%.0f)'
                      % (len(grip), grip.min(), grip.max(),
                         grip.min() * GRIPPER_SCALE,
                         grip.max() * GRIPPER_SCALE))
            ref = np.vstack([head, ref])
            print('  [执行] 位姿锚定: 推理后偏差 %.4f rad -> 用 %.2f s 平滑衔接'
                  % (gap, args.engage_time))
            print('  [执行] servoj %d 个 waypoint -> %d 条 @ %d Hz (%.2f s)'
                  % (plan.shape[0], ref.shape[0], SERVOJ_STREAM_HZ,
                     ref.shape[0] / SERVOJ_STREAM_HZ))
            stream_s, servo_stats = stream_trajectory(
                ws_servo, args.accid, ref, SERVOJ_STREAM_HZ,
                gripper=grip, sub_steps=sub_steps)
            time.sleep(0.2)
            after = read_robot_state(args.ws, args.accid)[0]
            print('     流 %.3f s (实测 %.0f Hz)；%s'
                  % (stream_s, ref.shape[0] / max(stream_s, 1e-9),
                     servo_summary(servo_stats)))
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
