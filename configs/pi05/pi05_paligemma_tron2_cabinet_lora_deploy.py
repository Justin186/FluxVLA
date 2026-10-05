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
"""Deployment config for the TRON2 cabinet PI0.5 LoRA policy.

Reuses the training config and overrides only ``inference_model`` with the
Triton + CUDA-Graph accelerated variant.

Measured on a 4090 D (sm_89), 3 camera views, 20 runs after warmup:

    PI05FlowMatching (plain)                236.3 ms
    PI05FlowMatchingRTCInference (accel.)    45.8 ms   (5.16x)

Notes
-----
1. ``PI05FlowMatchingRTCInference`` is required: the non-RTC
   ``PI05FlowMatchingInference`` raises
   ``RuntimeError: tensor a (2) vs b (3)`` with 3 camera views.
2. The first call costs ~29 s (Triton JIT + CUDA Graph capture). Warm the
   server up once before commanding the robot.
3. The speedup on sm_89 is 5.16x, not the ~15x quoted in
   ``docs/inference_acceleration.md`` (that figure is measured on A100).
"""

_base_ = './pi05_paligemma_tron2_cabinet_lora.py'

inference_model = dict(
    type='PI05FlowMatchingRTCInference',
    num_view=3,
    # Must cover the real prompt length: ``PreparePromptWithState`` writes all
    # 32 (padded) normalized state values into the prompt, which tokenizes to
    # ~135-142 tokens, and ``ProcessPrompts`` caps the prompt at 200.
    triton_max_prompt_len=200,
    num_steps=10,
    llm_backbone=dict(
        type='ConditionGemmaInferenceModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    vision_backbone=dict(
        type='SigLIPViTBackboneInference',
        vision_backbone_id='siglip_224',
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    projector=dict(
        type='LinearProjectorInference',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=50,
    action_in_proj=dict(
        type='LinearProjectorInference',
        in_dim=32,
        out_dim=1024,
    ),
    action_out_proj=dict(
        type='LinearProjectorInference',
        in_dim=1024,
        out_dim=32,
    ),
    time_mlp_in=dict(
        type='LinearProjectorInference',
        in_dim=1024,
        out_dim=1024,
    ),
    time_mlp_out=dict(
        type='LinearProjectorInference',
        in_dim=1024,
        out_dim=1024,
    ),
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaInferenceModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152,
    ),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path='./checkpoints/pi05_base_bf16/model.safetensors',
    name_mapping=dict(
        {
            'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
            'vision_backbone.vision': 'paligemma_with_expert.paligemma.model.vision_tower',
            'projector.projector': 'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
            'llm_expert': 'paligemma_with_expert.gemma_expert.model',
            'time_mlp_in.projector': 'time_mlp_in',
            'time_mlp_out.projector': 'time_mlp_out',
            'action_in_proj.projector': 'action_in_proj',
            'action_out_proj.projector': 'action_out_proj',
            'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
        }
    ),
    params_to_change_dtype=['llm_expert.llm.model.layers', 'vlm_backbone.vlm.model.language_model.layers', 'vlm_backbone.vlm.model.vision_tower', 'vlm_backbone.vlm.model.multi_modal_projector'],
    # Real (unpadded) action width.  Must match the training config, otherwise
    # the 32-dim padded prediction cannot be denormalized back to 16 dims.
    ori_action_dim=16,
)

# ZMQ serving wiring for ``scripts/zmq_inference_server.sh``.
# ``FluxVLAZMQEvalServer`` requires a ``themis`` section; without it the server
# aborts with ``KeyError: config.themis is required``.
#
# Observation contract of ``predict_action`` (batch size 1):
#   * ``qpos``   -- 16-dim state in *policy* order [L7, gripL, R7, gripR].
#                   It is the only field the dataset normalizes, and the
#                   quantile statistics are 16-dim, so it must stay 16-dim.
#   * ``states`` -- 18-dim raw robot state [L7, R7, head(2), gripL, gripR].
#                   DenormalizeDeltaAction uses it as the delta base and
#                   DenormalizeTron2Action reads head from indices 14-15.
#   * ``cam_high`` / ``cam_left_wrist`` / ``cam_right_wrist`` -- HxWx3 uint8.
#   * ``task_description`` -- task string, e.g. 'Press the red button'.
themis = dict(
    transport=dict(
        service_name='/fluxvla/predict_action',
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['qpos', 'states'],
        unnorm_key='private',
        image_encoding='rgb8',
    ),
    ros_server=dict(
        dataset_section='inference',
        device='cuda:0',
        # ⚠️ 必须打开，否则请求里的 seed 被完全忽略（ros_server.py:157/218）。
        # forward_seed=False 时 fluxvla 每次推理都用全局 RNG 现取的噪声，
        # 导致【同一个观测 + 同一个 seed 的两次调用可以差 0.086 rad】
        # （约合按钮间距的 1/3）—— 实测 10 次连续调用右臂逐点标准差 0.0106 rad。
        # 打开后：同 seed 可复现；客户端 --samples N 可用不同 seed 取平均，
        # 按 1/sqrt(N) 削掉这部分抖动。
        forward_seed=True,
    ),
)

