# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layout contract between the TRON2 policy and the TRON2 robot runner.

The cabinet policy emits 16 dims::

    [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]

while ``Tron2InferenceRunner`` and the robot consume 18 dims::

    [left_arm(7), right_arm(7), head(2), left_gripper(1), right_gripper(1)]

These tests pin the expansion, the raw-state permutation that must accompany
it, and the fact that the shipped config wires both up.
"""

from pathlib import Path

import numpy as np
import pytest
from mmengine import Config

from fluxvla.transforms.normalize import (DenormalizeDeltaAction,
                                          DenormalizeTron2Action)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / 'configs' / 'pi05' / 'pi05_paligemma_tron2_cabinet_lora.py')
STATS_PATH = (
    REPO_ROOT / 'datasets' / 'RealRobot_Tron2_lerobot' /
    'tron2_stats_armsymmetric.json')

POLICY_DIM = 16
ROBOT_DIM = 18
# Robot order -> policy order: [L7, R7, head2, gripL, gripR]
#                            -> [L7, gripL, R7, gripR, head2]
STATE_PERMUTATION = [
    0, 1, 2, 3, 4, 5, 6, 16, 7, 8, 9, 10, 11, 12, 13, 17, 14, 15
]
DELTA_MASK = [True] * 7 + [False] + [True] * 7 + [False]
# Slots the runner slices when commanding the robot.
RUNNER_LEFT_ARM = slice(0, 7)
RUNNER_RIGHT_ARM = slice(7, 14)
RUNNER_HEAD = slice(14, 16)
RUNNER_LEFT_GRIPPER = 16
RUNNER_RIGHT_GRIPPER = 17


def build_transform(kind=DenormalizeTron2Action, **overrides):
    kwargs = dict(
        delta_action_mask=DELTA_MASK,
        norm_type='quantile',
        action_dim=POLICY_DIM,
        norm_stats=str(STATS_PATH),
        state_permutation=STATE_PERMUTATION)
    kwargs.update(overrides)
    return kind(**kwargs)


def robot_state():
    """Native 18-dim robot state: [L(7), R(7), head(2), gripL, gripR]."""
    return np.arange(ROBOT_DIM, dtype=np.float32)


def delta_contribution(transform, num_steps=4):
    """Which robot-state value each output slot received.

    The delta offset is additive and applied after denormalization, so
    differencing against a zero state isolates it exactly.
    """
    action = np.zeros((num_steps, POLICY_DIM), dtype=np.float32)
    zero = transform(dict(action=action.copy(), state=np.zeros(ROBOT_DIM)))
    moved = transform(dict(action=action.copy(), state=robot_state()))
    return np.asarray(moved)[0] - np.asarray(zero)[0]


def test_expands_policy_layout_into_robot_layout():
    expanded = build_transform()(
        dict(
            action=np.zeros((4, POLICY_DIM), dtype=np.float32),
            state=np.zeros(ROBOT_DIM)))
    assert np.asarray(expanded).shape == (4, ROBOT_DIM)


def test_delta_state_lands_in_robot_slots():
    contribution = delta_contribution(build_transform())
    # Left arm: policy 0:7 -> robot 0:7.
    assert contribution[RUNNER_LEFT_ARM] == pytest.approx(robot_state()[:7])
    # Right arm: policy 8:15 -> robot 7:14. Without the permutation this
    # receives robot indices 8..14, i.e. a right arm shifted by one joint
    # plus head_pitch.
    assert contribution[RUNNER_RIGHT_ARM] == pytest.approx(robot_state()[7:14])
    # Grippers are absolute, so no delta is added to their robot slots.
    assert contribution[RUNNER_LEFT_GRIPPER] == pytest.approx(0.0)
    assert contribution[RUNNER_RIGHT_GRIPPER] == pytest.approx(0.0)


def test_head_slots_repeat_the_current_head_position():
    contribution = delta_contribution(build_transform())
    assert contribution[RUNNER_HEAD] == pytest.approx(robot_state()[14:16])


def test_head_slots_default_to_zero_without_a_usable_state():
    transform = build_transform()
    assert transform.current_head({}) == pytest.approx([0.0, 0.0])
    short_state = np.zeros(ROBOT_DIM - 1, dtype=np.float32)
    assert transform.current_head(dict(state=short_state)) == pytest.approx(
        [0.0, 0.0])


def test_gripper_slots_follow_the_policy_order():
    # Symmetric stats make denormalization an identity map.  The collected
    # left-gripper dimension is identically zero, so its real quantile range
    # is [0, 0] and every input collapses to 0, which would hide a swap.
    width = POLICY_DIM
    block = dict(
        q01=[-1.0] * width,
        q99=[1.0] * width,
        mean=[0.0] * width,
        std=[1.0] * width,
        min=[-1.0] * width,
        max=[1.0] * width,
        count=1)
    stats = dict(private=dict(action=dict(block), proprio=dict(block)))
    transform = build_transform(norm_stats=stats)

    # Two steps: the parent squeezes a single-step 2-D action to 1-D.
    action = np.zeros((2, POLICY_DIM), dtype=np.float32)
    action[:, 7] = 0.25  # left gripper
    action[:, 15] = 0.75  # right gripper
    out = np.asarray(transform(dict(action=action, state=np.zeros(ROBOT_DIM))))
    assert out[0, RUNNER_LEFT_GRIPPER] == pytest.approx(0.25)
    assert out[0, RUNNER_RIGHT_GRIPPER] == pytest.approx(0.75)


def test_unexpanded_action_cannot_be_indexed_by_the_runner():
    """Regression guard: the plain delta transform is not enough."""
    plain = np.asarray(
        build_transform(kind=DenormalizeDeltaAction)(dict(
            action=np.zeros((2, POLICY_DIM), dtype=np.float32),
            state=robot_state())))
    assert plain.shape[-1] == POLICY_DIM
    with pytest.raises(IndexError):
        plain[:, RUNNER_LEFT_GRIPPER]


def test_permutation_must_cover_every_index():
    # A 16-long permutation is the tempting mistake: state_permutation
    # reorders within the raw width, it cannot drop dimensions.
    with pytest.raises(ValueError, match='exactly once'):
        build_transform(state_permutation=[
            0, 1, 2, 3, 4, 5, 6, 16, 7, 8, 9, 10, 11, 12, 13, 17
        ])


def test_rejects_unexpected_action_width():
    transform = build_transform()
    with pytest.raises(ValueError, match='already expanded'):
        transform(
            dict(
                action=np.zeros((2, 15), dtype=np.float32),
                state=np.zeros(ROBOT_DIM)))


def test_already_expanded_action_passes_through():
    transform = build_transform()
    robot_action = np.arange(2 * ROBOT_DIM, dtype=np.float32).reshape(2, -1)
    out = np.asarray(
        transform(dict(action=robot_action, state=np.zeros(ROBOT_DIM))))
    assert out.shape == robot_action.shape


def test_action_dim_is_required():
    with pytest.raises(ValueError, match='action_dim'):
        build_transform(action_dim=None)


def test_shipped_config_matches_the_tested_layout():
    cfg = Config.fromfile(str(CONFIG_PATH))
    denorm = cfg.inference.denormalize_action
    assert denorm['type'] == 'DenormalizeTron2Action'
    assert denorm['action_dim'] == POLICY_DIM
    assert denorm['delta_action_mask'] == DELTA_MASK
    assert sorted(denorm['state_permutation']) == list(range(ROBOT_DIM))
    # The shipped values must produce the same mapping asserted above.
    transform = build_transform(state_permutation=denorm['state_permutation'])
    assert delta_contribution(transform)[RUNNER_RIGHT_ARM] == pytest.approx(
        robot_state()[7:14])
