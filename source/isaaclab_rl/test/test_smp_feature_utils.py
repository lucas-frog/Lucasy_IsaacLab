# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import math
from pathlib import Path

import pytest
import torch


def _load_smp_features_module():
    # 直接按文件路径加载特征工具模块，避免测试依赖安装路径。
    module_path = (
        Path(__file__).resolve().parents[2]
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "mdp"
        / "smp_features.py"
    )
    spec = importlib.util.spec_from_file_location("isaaclab_smp_feature_utils_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_g1_smp_config_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "g1"
        / "agents"
        / "config.py"
    )
    spec = importlib.util.spec_from_file_location("isaaclab_g1_smp_config_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _identity_key_body_quat(*lead_shape: int, num_bodies: int = 14) -> torch.Tensor:
    # 生成单位四元数输入，作为旋转相关特征测试的稳定基线。
    quat = torch.zeros(*lead_shape, num_bodies, 4, dtype=torch.float32)
    quat[..., 0] = 1.0
    return quat


def test_quat_to_rot6d_returns_six_values_per_body():
    # 验证 quat->rot6d 转换后每个刚体输出 6 维表示。
    smp_features = _load_smp_features_module()
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    rot6d = smp_features.quat_to_rot6d(quat)

    assert rot6d.shape == (1, 1, 6)
    assert torch.allclose(rot6d, torch.tensor([[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]]))


def test_quat_to_rot6d_preserves_row_major_first_two_columns_order():
    # 验证 rot6d 的展开顺序与约定一致（行优先的前两列）。
    smp_features = _load_smp_features_module()
    half_sqrt = math.sqrt(0.5)
    quat = torch.tensor([[[half_sqrt, 0.0, 0.0, half_sqrt]]], dtype=torch.float32)

    rot6d = smp_features.quat_to_rot6d(quat)

    expected = torch.tensor([[[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(rot6d, expected, atol=1e-5)


def test_heading_frame_maps_world_up_to_local_y_axis():
    # heading-local 坐标系要求 local y 与全局 up 对齐。
    smp_features = _load_smp_features_module()
    rotation = smp_features.build_heading_frame_rotation(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))

    local_up = smp_features.world_to_local_frame(
        rotation,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )

    assert torch.allclose(local_up, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32), atol=1e-5)


def test_joint_angle_offsets_to_rot6d_uses_relative_default_pose():
    # 单关节 rot6d 应编码 joint_pos - default_joint_pos 的相对旋转。
    smp_features = _load_smp_features_module()

    rot6d = smp_features.joint_angle_offsets_to_rot6d(
        torch.tensor([[math.pi / 2]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )

    expected = torch.tensor([[[0.0, 0.0, 0.0, 1.0, -1.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(rot6d, expected, atol=1e-5)


def test_joint_rot6d_roundtrip_recovers_single_axis_offsets():
    # GSI 需要把 joint rot6d 反解为单轴关节角偏移。
    smp_features = _load_smp_features_module()
    offsets = torch.tensor([[0.3, -0.5]], dtype=torch.float32)
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    rot6d = smp_features.joint_angle_offsets_to_rot6d(offsets, axes)
    recovered = smp_features.joint_rot6d_to_angle_offsets(rot6d, axes)

    assert torch.allclose(recovered, offsets, atol=1e-5)


def test_pack_smp_frame_features_has_expected_dim():
    # 验证单帧特征打包后的最终维度与设计值一致。
    smp_features = _load_smp_features_module()

    features = smp_features.pack_smp_frame_features(
        base_lin_vel_b=torch.zeros(2, 3),
        base_ang_vel_b=torch.zeros(2, 3),
        joint_rot6d_rel=torch.zeros(2, 29, 6),
        ee_pos_b=torch.zeros(2, 4, 3),
    )

    assert features.shape == (2, 192)


def test_pack_smp_frame_features_raises_for_unexpected_dim():
    # 维度约束回归：当期望维度不匹配时应抛出明确异常。
    smp_features = _load_smp_features_module()

    with pytest.raises(ValueError, match="Expected SMP feature dim 191, got 192"):
        smp_features.pack_smp_frame_features(
            base_lin_vel_b=torch.zeros(2, 3),
            base_ang_vel_b=torch.zeros(2, 3),
            joint_rot6d_rel=torch.zeros(2, 29, 6),
            ee_pos_b=torch.zeros(2, 4, 3),
            expected_feature_dim=191,
        )


def test_pack_smp_frame_features_supports_multiple_leading_dims():
    # 验证函数可处理额外前导维（如 batch+time）并保持末维为特征维。
    smp_features = _load_smp_features_module()

    features = smp_features.pack_smp_frame_features(
        base_lin_vel_b=torch.zeros(2, 5, 3),
        base_ang_vel_b=torch.zeros(2, 5, 3),
        joint_rot6d_rel=torch.zeros(2, 5, 29, 6),
        ee_pos_b=torch.zeros(2, 5, 4, 3),
        expected_feature_dim=192,
    )

    assert features.shape == (2, 5, 192)


def test_pack_smp_frame_features_supports_extended_198_schema():
    # 新 schema 应保留旧 192D 前缀，并在末尾追加 world-frame 根速度。
    smp_features = _load_smp_features_module()

    features = smp_features.pack_smp_frame_features(
        base_lin_vel_b=torch.zeros(2, 3),
        base_ang_vel_b=torch.zeros(2, 3),
        joint_rot6d_rel=torch.zeros(2, 29, 6),
        ee_pos_b=torch.zeros(2, 4, 3),
        base_lin_vel_w=torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        base_ang_vel_w=torch.tensor([[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]]),
        feature_schema="extended_198",
        expected_feature_dim=198,
    )

    assert features.shape == (2, 198)
    assert torch.allclose(features[0, -6:], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.allclose(features[1, -6:], torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))


def test_base_velocity_command_to_world_velocities_uses_root_heading():
    # base_velocity command 是 base-frame [vx, vy, yaw_rate]，extended_198 末尾需要 world-frame 6D。
    smp_features = _load_smp_features_module()
    half_sqrt = math.sqrt(0.5)

    lin_vel_w, ang_vel_w = smp_features.base_velocity_command_to_world_velocities(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [half_sqrt, 0.0, 0.0, half_sqrt],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, -0.5],
            ],
            dtype=torch.float32,
        ),
    )

    assert torch.allclose(lin_vel_w[0], torch.tensor([1.0, 2.0, 0.0]), atol=1e-5)
    assert torch.allclose(ang_vel_w[0], torch.tensor([0.0, 0.0, 3.0]), atol=1e-5)
    assert torch.allclose(lin_vel_w[1], torch.tensor([-2.0, 1.0, 0.0]), atol=1e-5)
    assert torch.allclose(ang_vel_w[1], torch.tensor([0.0, 0.0, -0.5]), atol=1e-5)


def test_g1_smp_config_computes_feature_layout_by_schema():
    g1_config = _load_g1_smp_config_module()

    assert g1_config.g1_smp_feature_dim_for_schema("legacy_192") == 192
    assert g1_config.g1_smp_feature_dim_for_schema("extended_198") == 198
    assert g1_config.g1_smp_feature_block_offsets_for_schema("extended_198")["base_lin_vel_w"] == (192, 195)
    assert g1_config.g1_smp_feature_block_offsets_for_schema("extended_198")["base_ang_vel_w"] == (195, 198)
