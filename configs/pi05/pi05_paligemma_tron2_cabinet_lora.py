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

model = dict(
    type='PI05FlowMatching',
    llm_backbone=dict(
        type='ConditionGemmaModel',
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
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
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
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=50,
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    # Match the OpenPI-aligned RoboCasa flow-matching objective.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=False,
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
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
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    use_lora=True,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.0,
    lora_target_modules=[
        'q_proj',
        'v_proj',
        'k_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
        'projector.projector',
        'out_proj',
        'fc1',
        'fc2',
    ],
    modules_to_save=[
        'action_in_proj',
        'action_out_proj',
        'time_mlp_in',
        'time_mlp_out',
    ],
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/pi05_base_bf16/model.safetensors',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=16,
    # Supervise all padded model dimensions, as in OpenPI.
    loss_action_dim=16,
)

inference_model = model.copy()

train_dataloader = dict(
    # micro-batch=3 是本卡(4090D 24G)上限：4 会 OOM，1/2 则 GPU 利用率低。
    # 实测每样本耗时 batch1=0.34s / batch2=0.219s / batch3=0.179s。
    per_device_batch_size=3,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        # [left arm (7), left gripper, right arm (7), right gripper].
        # Arm joints use state-relative deltas; grippers remain absolute.
        #
        # 不用自动统计，改用人工修正版（原因见文件内 _fix_note）：
        # 左臂在全部 5 个数据集里全程静止，其相对动作的 q01-q99 跨度塌缩到
        # ~0.006，quantile 归一化 (x-q01)/(q99-q01)*2-1 把噪声放大约 150 倍
        # → 左臂占了 67.5% 的 loss（且这 67.5% 是不可预测的噪声），
        # 并制造出最高 629 的 loss 尖峰；而真正干活的右臂只占 29.7%。
        # 修正：左臂 0-6 维改用对称右臂 8-14 维的 q01/q99（同名关节运动范围
        # 相同），放大倍数从 130~194 降到 0.51~1.30。
        dataset_statistics_path=  # noqa: E251
        './datasets/RealRobot_Tron2_lerobot/tron2_stats_armsymmetric.json',  # noqa: E501
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action'],
        datasets=[
            dict(
                type='ParquetDatasetV3',
                data_root_path=[  # noqa: E501
                './datasets/RealRobot_Tron2_lerobot/lerobot_2026-10-02_22-55-58',  # noqa: E501
                './datasets/RealRobot_Tron2_lerobot/lerobot_2026-10-02_23-24-38',  # noqa: E501
                './datasets/RealRobot_Tron2_lerobot/lerobot_2026-10-02_23-53-41',  # noqa: E501
            ],
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist'
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        }),
                    dict(
                        type='RelativeActions',
                        mask=[True] * 7 + [False] + [True] * 7 + [False]),
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=32,
                        state_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='quantile',
                        output_dtype='float32'),
                    dict(type='PreparePromptWithState'),
                    dict[str, str | dict[str, str]](
                        type='ProcessPrompts',
                        max_len=200,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'checkpoints/pi05_base',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(
                        type='ResizeImagesWithPad',
                        height=224,
                        width=224,
                        backend='pil'),
                    dict(type='SimpleNormalizeImages'),
                    dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
                ],
                action_key='action',
                window_start_idx=0,
                action_window_size=50,
                supervise_terminal_padding=True)
        ]))

runner = dict(
    type='DDPTrainRunner',
    max_epochs=None,
    # 56485 帧 x 7.2 epoch / 24 = 17000 步，按实测约 20 小时。
    # 用 17000 而非 16600 是为了能被 save_iter_interval 整除：
    # 步骤态下只有 save_iter_interval 一条保存路径，且训练循环结束时
    # 【没有】收尾保存，不整除则最后一步不会落盘。
    max_steps=25000,
    # 每 1000 步滚动保存。注意单份检查点约 29.2 GB
    # (.pt 14.75 GB + .safetensors 14.47 GB，两者内容重复)。
    save_iter_interval=1000,
    # 保存时先写新的再删旧的，峰值多占一份：
    # 峰值 (3+1) x 29.2 = 116.8 GB，磁盘仅余约 29 GB，不可再调大。
    max_keep_ckpts=3,
    # 3 samples/GPU x 8 accumulation steps = effective batch 24。
    # 厂商原版是 8 samples/GPU x 4 GPUs x 2 accum = batch 64，
    # 单卡 24G 装不下 micro-batch 8，故用累积等效缩小。
    grad_accumulation_steps=16,
    seed=42,
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    # BF16 compute with FP32 sharded master parameters and reductions.
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        warmup_steps=1000,
        # 与 max_steps 一致，让余弦退火在训练结束时刚好收敛到 min_lr
        # （完整退火），比厂商"decay_steps > max_steps"的半程退火更友好。
        decay_steps=25000,
        min_lr=1e-6),
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'checkpoints/pi05_base',  # noqa: E501
        # special_tokens={'pad_token': '<PAD>'}
    ),
    metric=dict(
        type='VLAMetric',
        # jsonl 保持每步一条；csv 每 100 步一行，整行包含 push() 的全部指标
        # （loss / loss_raw / l1_loss / action_accuracy / lr / step_time 等）。
        active_trackers=('jsonl', 'csv'),
        csv_interval=100,
        run_dir='work_dirs',
        window_size=1),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=False,
    static_graph=False)

inference = dict(
    type='Tron2InferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    # ⚠️ 必须与训练数据集 meta/tasks.parquet 里的任务文本逐字一致：
    #   lerobot_2026-10-02_22-55-58 -> 'Press the red button'
    #   lerobot_2026-10-02_23-24-38 -> 'Press the black button'
    #   lerobot_2026-10-02_23-53-41 -> 'Press the green button'
    # PreparePromptWithState 的 lowercase_task_description 默认 False，
    # 只做 strip()/_→空格/去换行，所以大小写与措辞一个字都不能改。
    task_descriptions={
        '1': 'Press the red button',
        '2': 'Press the black button',
        '3': 'Press the green button',
    },
    seed=7,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_dim=32,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(type='PreparePromptWithState'),
            dict[str, str | dict[str, str]](
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    'checkpoints/pi05_base',
                    # special_tokens={'pad_token': '<PAD>'}
                )),
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages'),
        ]),
    # 模型输出 16 维 [L7,gripL,R7,gripR]，而 Tron2InferenceRunner
    # 和机器人需要 18 维 [L7,R7,head2,gripL,gripR]。展开放在这里，
    # 本机直跑 (BaseInferenceRunner._postprocess_actions) 与 ZMQ
    # 服务两条路径都会经过 denormalize_action，一处覆盖两条。
    denormalize_action=dict(
        type='DenormalizeTron2Action',
        norm_type='quantile',
        action_dim=16,
        delta_action_mask=[True] * 7 + [False] + [True] * 7 + [False],
        # 机器人原始状态 18 维 [L7,R7,head2,gripL,gripR]，要重排成
        # 模型动作顺序 [L7,gripL,R7,gripR,head2]。前 16 位被
        # delta_action_mask 使用，末尾两位（头部）不参与。
        # state_permutation 只能重排不能筛选，所以长度必须是 18；
        # 写成 16 会触发 "must contain every index in [0, D)"。
        state_permutation=[0, 1, 2, 3, 4, 5, 6, 16, 7, 8, 9, 10, 11, 12,
                           13, 17, 14, 15],
    ),
    action_chunk=32,
    operator=dict(
        type='Tron2Operator',
        image_encoding='rgb8',
        img_left_topic='/camera/left/color/image_rect_raw',
        img_right_topic='/camera/right/color/image_rect_raw',
        img_top_topic='/camera/top/color/image_raw',
        joint_state_topic='/joint_states',
        gripper_state_topic='/gripper_state',
        ee_pose_left_topic='/left_arm/ee_pose',
        ee_pose_right_topic='/right_arm/ee_pose',
    ))
