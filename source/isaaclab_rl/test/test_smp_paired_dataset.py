import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_paired_dataset_module():
    module_path = Path(__file__).resolve().parents[1] / "isaaclab_rl" / "rsl_rl" / "smp_paired_dataset.py"
    spec = importlib.util.spec_from_file_location("isaaclab_smp_paired_dataset_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _raw_states(num_frames: int = 5) -> dict[str, np.ndarray]:
    joint_pos = np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3) * 0.01
    joint_vel = np.ones_like(joint_pos) * 0.5
    body_pos_w = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_pos_w[:, 0, 0] = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    body_quat_w = np.zeros((num_frames, 2, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, 2, 3), dtype=np.float32)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def test_save_paired_dataset_writes_training_frames_raw_states_and_offsets(tmp_path):
    module = _load_paired_dataset_module()
    output_path = tmp_path / "paired_walk.npz"
    frames = np.arange(20, dtype=np.float32).reshape(5, 4)
    feature_block_offsets = {
        "base_lin_vel_b": (0, 1),
        "base_ang_vel_b": (1, 2),
        "joint_rot6d_rel": (2, 4),
    }

    module.save_smp_paired_dataset(
        output_path,
        frames=frames,
        fps=50,
        raw_states=_raw_states(),
        joint_names=["joint_a", "joint_b", "joint_c"],
        joint_axes=np.eye(3, dtype=np.float32),
        ee_names=["left_foot", "right_foot"],
        feature_block_offsets=feature_block_offsets,
        window_size=10,
        stride=1,
        style_name="walk",
        style_id=0,
        source_name="walk1",
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert np.array_equal(data["frames"], frames)
        assert int(data["feature_dim"][0]) == 4
        assert int(data["window_size"][0]) == 10
        assert list(data["feature_block_names"]) == list(feature_block_offsets)
        assert np.array_equal(data["feature_block_offsets"], np.array([[0, 1], [1, 2], [2, 4]]))
        assert list(data["joint_names"]) == ["joint_a", "joint_b", "joint_c"]
        assert str(data["style_name"][0]) == "walk"
        assert int(data["style_id"][0]) == 0
        assert str(data["source_name"][0]) == "walk1"
        assert data["joint_pos"].shape == (5, 3)
        assert data["body_pos_w"].shape == (5, 2, 3)


def test_save_paired_dataset_writes_feature_schema_metadata(tmp_path):
    module = _load_paired_dataset_module()
    output_path = tmp_path / "paired_walk_198.npz"
    frames = np.zeros((5, 10), dtype=np.float32)

    module.save_smp_paired_dataset(
        output_path,
        frames=frames,
        fps=50,
        raw_states=_raw_states(),
        joint_names=["joint_a", "joint_b", "joint_c"],
        joint_axes=np.eye(3, dtype=np.float32),
        ee_names=[],
        feature_block_offsets={"base_lin_vel_b": (0, 3), "base_ang_vel_b": (3, 6), "joint_rot6d_rel": (6, 10)},
        feature_schema="extended_198",
        window_size=10,
        stride=1,
    )

    with np.load(output_path, allow_pickle=False) as data:
        assert str(data["feature_schema"][0]) == "extended_198"


def test_read_feature_schema_infers_legacy_for_old_192_payloads():
    module = _load_paired_dataset_module()

    assert module.read_feature_schema({"frames": np.zeros((5, 192), dtype=np.float32)}) == "legacy_192"


def test_save_paired_dataset_stores_distinct_raw_and_smp_joint_orders(tmp_path):
    module = _load_paired_dataset_module()
    output_path = tmp_path / "paired_ordered.npz"

    module.save_smp_paired_dataset(
        output_path,
        frames=np.zeros((5, 8), dtype=np.float32),
        fps=50,
        raw_states=_raw_states(),
        joint_names=["sim_joint_c", "sim_joint_a", "sim_joint_b"],
        joint_axes=np.eye(3, dtype=np.float32),
        ee_names=[],
        feature_block_offsets={"base_lin_vel_b": (0, 3), "base_ang_vel_b": (3, 6), "joint_rot6d_rel": (6, 8)},
        window_size=2,
        stride=1,
        smp_joint_names=["sim_joint_a", "sim_joint_b", "sim_joint_c"],
        body_names=["pelvis", "left_foot"],
    )

    payload = module.load_smp_paired_dataset(output_path)

    assert module.read_replay_joint_names(payload) == ["sim_joint_c", "sim_joint_a", "sim_joint_b"]
    assert list(payload["smp_joint_names"]) == ["sim_joint_a", "sim_joint_b", "sim_joint_c"]
    assert list(payload["body_names"]) == ["pelvis", "left_foot"]


def test_validate_smp_frames_match_reports_overall_and_block_errors():
    module = _load_paired_dataset_module()
    expected = torch.zeros((3, 6), dtype=torch.float32)
    actual = expected.clone()
    actual[:, 0] = 0.5
    actual[:, 3:6] = 2.0

    result = module.validate_smp_frames_match(
        actual,
        expected,
        feature_block_offsets={
            "base_lin_vel_b": (0, 3),
            "base_ang_vel_b": (3, 6),
        },
    )

    assert result.max_abs_error == pytest.approx(2.0)
    assert result.mean_abs_error == pytest.approx((3 * 0.5 + 9 * 2.0) / 18)
    assert result.block_errors["base_lin_vel_b"].max_abs_error == pytest.approx(0.5)
    assert result.block_errors["base_ang_vel_b"].max_abs_error == pytest.approx(2.0)


def test_load_paired_raw_state_clip_slices_by_fps_and_duration(tmp_path):
    module = _load_paired_dataset_module()
    output_path = tmp_path / "paired_walk.npz"
    frames = np.zeros((6, 4), dtype=np.float32)
    raw_states = _raw_states(num_frames=6)

    module.save_smp_paired_dataset(
        output_path,
        frames=frames,
        fps=10,
        raw_states=raw_states,
        joint_names=["joint_a", "joint_b", "joint_c"],
        joint_axes=np.eye(3, dtype=np.float32),
        ee_names=[],
        feature_block_offsets={"base_lin_vel_b": (0, 1), "base_ang_vel_b": (1, 2), "joint_rot6d_rel": (2, 4)},
        window_size=2,
        stride=1,
    )

    clip = module.load_paired_raw_state_clip(output_path, clip_start_sec=0.2, clip_duration_sec=0.3)

    assert clip.fps == pytest.approx(10.0)
    assert clip.frame_range == (2, 5)
    assert torch.equal(clip.joint_pos, torch.as_tensor(raw_states["joint_pos"][2:5]))
    assert torch.equal(clip.root_pos_w, torch.as_tensor(raw_states["body_pos_w"][2:5, 0]))


def test_should_write_full_joint_state_detects_full_width_raw_vectors():
    module = _load_paired_dataset_module()

    assert module.should_write_full_joint_state(joint_state_width=29, robot_joint_count=29) is True
    assert module.should_write_full_joint_state(joint_state_width=12, robot_joint_count=29) is False


def test_build_g1_smp_frames_from_raw_motion_encodes_identity_joint_offsets():
    module = _load_paired_dataset_module()
    raw_motion = {
        "joint_pos": np.zeros((2, 1), dtype=np.float32),
        "body_pos_w": np.zeros((2, 2, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 2, 1)),
        "body_lin_vel_w": np.zeros((2, 2, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((2, 2, 3), dtype=np.float32),
    }
    raw_motion["body_pos_w"][:, 1, 0] = 0.25

    frames = module.build_g1_smp_frames_from_raw_motion(
        raw_motion,
        body_order=["pelvis", "left_foot"],
        joint_names=["hip_pitch_joint"],
        joint_axes=[(0.0, 1.0, 0.0)],
        default_joint_pos=np.zeros((1,), dtype=np.float32),
        ee_names=["left_foot"],
        expected_feature_dim=15,
    )

    assert frames.shape == (2, 15)
    assert torch.allclose(frames[:, 6:12], torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]).repeat(2, 1))
    assert torch.allclose(frames[:, 12:15], torch.tensor([[0.25, 0.0, 0.0]]).repeat(2, 1))


def test_g1_smp_feature_block_offsets_support_extended_schema():
    module = _load_paired_dataset_module()

    offsets = module.g1_smp_feature_block_offsets(1, 1, feature_schema="extended_198")

    assert offsets == {
        "base_lin_vel_b": (0, 3),
        "base_ang_vel_b": (3, 6),
        "joint_rot6d_rel": (6, 12),
        "ee_pos_b": (12, 15),
        "base_lin_vel_w": (15, 18),
        "base_ang_vel_w": (18, 21),
    }


def test_build_g1_smp_frames_from_raw_motion_supports_extended_schema():
    module = _load_paired_dataset_module()
    raw_motion = {
        "joint_pos": np.zeros((2, 1), dtype=np.float32),
        "body_pos_w": np.zeros((2, 2, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 2, 1)),
        "body_lin_vel_w": np.tile(np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]], dtype=np.float32), (2, 1, 1)),
        "body_ang_vel_w": np.tile(np.array([[[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]], dtype=np.float32), (2, 1, 1)),
    }
    raw_motion["body_pos_w"][:, 1, 0] = 0.25

    frames = module.build_g1_smp_frames_from_raw_motion(
        raw_motion,
        body_order=["pelvis", "left_foot"],
        joint_names=["hip_pitch_joint"],
        joint_axes=[(0.0, 1.0, 0.0)],
        default_joint_pos=np.zeros((1,), dtype=np.float32),
        ee_names=["left_foot"],
        expected_feature_dim=21,
        feature_schema="extended_198",
    )

    assert frames.shape == (2, 21)
    assert torch.allclose(frames[:, -6:], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).repeat(2, 1))
