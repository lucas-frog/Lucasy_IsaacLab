# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


def _load_module(module_name: str, module_path: Path):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_smp_denoising_diagnostic_module():
    for parent in Path(__file__).resolve().parents:
        package_root = parent / "isaaclab_rl"
        rsl_rl_root = package_root / "rsl_rl"
        module_path = rsl_rl_root / "smp_denoising_diagnostic.py"
        if rsl_rl_root.exists():
            package = types.ModuleType("isaaclab_rl")
            package.__path__ = [str(package_root)]
            sys.modules["isaaclab_rl"] = package

            subpackage = types.ModuleType("isaaclab_rl.rsl_rl")
            subpackage.__path__ = [str(rsl_rl_root)]
            sys.modules["isaaclab_rl.rsl_rl"] = subpackage

            _load_module("isaaclab_rl.rsl_rl.smp_paired_dataset", rsl_rl_root / "smp_paired_dataset.py")
            _load_module("isaaclab_rl.rsl_rl.smp_denoised_playback", rsl_rl_root / "smp_denoised_playback.py")
            return _load_module("isaaclab_rl.rsl_rl.smp_denoising_diagnostic", module_path)
    raise FileNotFoundError("Could not find isaaclab_rl/rsl_rl/smp_denoising_diagnostic.py")


def _make_toy_paired_dataset(path: Path) -> Path:
    num_frames = 8
    frames = np.zeros((num_frames, 12), dtype=np.float32)
    frames[:, 6:12] = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    payload = {
        "frames": frames,
        "fps": np.asarray([50.0], dtype=np.float32),
        "window_size": np.asarray([4], dtype=np.int64),
        "stride": np.asarray([2], dtype=np.int64),
        "feature_dim": np.asarray([12], dtype=np.int64),
        "joint_names": np.asarray(["raw_joint_a", "raw_joint_b"], dtype=np.str_),
        "smp_joint_names": np.asarray(["smp_joint"], dtype=np.str_),
        "joint_axes": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        "ee_names": np.asarray([], dtype=np.str_),
        "feature_block_names": np.asarray(["base_lin_vel_b", "base_ang_vel_b", "joint_rot6d_rel"], dtype=np.str_),
        "feature_block_offsets": np.asarray([[0, 3], [3, 6], [6, 12]], dtype=np.int64),
        "joint_pos": np.zeros((num_frames, 2), dtype=np.float32),
        "joint_vel": np.zeros((num_frames, 2), dtype=np.float32),
        "body_pos_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, 1, 1)),
        "body_lin_vel_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "source_name": np.asarray(["toy_clip"], dtype=np.str_),
    }
    np.savez(path, **payload)
    return path


def _make_toy_paired_dataset_with_moving_raw_root(path: Path) -> Path:
    num_frames = 8
    frames = np.zeros((num_frames, 12), dtype=np.float32)
    frames[:, 6:12] = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    body_pos_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    body_pos_w[:, 0, 0] = np.linspace(10.0, 10.7, num_frames, dtype=np.float32)
    body_pos_w[:, 0, 1] = 20.0
    body_pos_w[:, 0, 2] = 1.0

    payload = {
        "frames": frames,
        "fps": np.asarray([50.0], dtype=np.float32),
        "window_size": np.asarray([4], dtype=np.int64),
        "stride": np.asarray([2], dtype=np.int64),
        "feature_dim": np.asarray([12], dtype=np.int64),
        "joint_names": np.asarray(["raw_joint_a", "raw_joint_b"], dtype=np.str_),
        "smp_joint_names": np.asarray(["smp_joint"], dtype=np.str_),
        "joint_axes": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        "ee_names": np.asarray([], dtype=np.str_),
        "feature_block_names": np.asarray(["base_lin_vel_b", "base_ang_vel_b", "joint_rot6d_rel"], dtype=np.str_),
        "feature_block_offsets": np.asarray([[0, 3], [3, 6], [6, 12]], dtype=np.int64),
        "joint_pos": np.zeros((num_frames, 2), dtype=np.float32),
        "joint_vel": np.zeros((num_frames, 2), dtype=np.float32),
        "body_pos_w": body_pos_w,
        "body_quat_w": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, 1, 1)),
        "body_lin_vel_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "source_name": np.asarray(["toy_clip"], dtype=np.str_),
    }
    np.savez(path, **payload)
    return path


def test_select_playback_states_uses_raw_clip_for_raw_mode():
    module = _load_smp_denoising_diagnostic_module()

    raw_clip = types.SimpleNamespace(playback_states="raw-state")

    selected = module.select_playback_states(play_source="raw", raw_clip=raw_clip, decoded_state_map={})

    assert selected is raw_clip


def test_select_playback_joint_names_uses_decoded_names_for_decoded_mode():
    module = _load_smp_denoising_diagnostic_module()

    selected = module.select_playback_joint_names(
        play_source="denoised_decoded",
        raw_joint_names=("raw_joint_a", "raw_joint_b"),
        decoded_joint_names=("smp_joint",),
    )

    assert selected == ("smp_joint",)


def test_should_use_direct_joint_write_requires_width_and_name_match():
    module = _load_smp_denoising_diagnostic_module()

    assert module.should_use_direct_joint_write(
        joint_state_width=2,
        robot_joint_names=("joint_a", "joint_b"),
        playback_joint_names=("joint_a", "joint_b"),
    )
    assert not module.should_use_direct_joint_write(
        joint_state_width=2,
        robot_joint_names=("joint_a", "joint_b"),
        playback_joint_names=("joint_b", "joint_a"),
    )


def test_should_use_direct_joint_write_keeps_stable_raw_full_vector_policy():
    module = _load_smp_denoising_diagnostic_module()

    assert module.should_use_direct_joint_write(
        joint_state_width=2,
        robot_joint_names=("joint_a", "joint_b"),
        playback_joint_names=("joint_b", "joint_a"),
        play_source="raw",
    )


def test_should_use_direct_joint_write_accepts_paired_raw_reference_order():
    module = _load_smp_denoising_diagnostic_module()

    assert module.should_use_direct_joint_write(
        joint_state_width=2,
        robot_joint_names=("robot_joint_b", "robot_joint_a"),
        playback_joint_names=("joint_a", "joint_b"),
        reference_joint_names=("joint_a", "joint_b"),
        play_source="clean_decoded",
    )


def test_prepare_diagnostic_clip_returns_clean_noisy_and_denoised_frames(tmp_path, monkeypatch):
    module = _load_smp_denoising_diagnostic_module()
    dataset_path = _make_toy_paired_dataset(tmp_path / "dataset_walk.npz")

    class _DummyScheduler:
        def __init__(self):
            self.alpha_bar = torch.linspace(1.0, 0.1, 10, dtype=torch.float32)

        def q_sample(self, windows, timesteps, noise):
            del timesteps
            return windows + 0.25 * noise

    class _DummyModel:
        num_styles = 0

        def __call__(self, xt, timesteps, style_id=None):
            del timesteps, style_id
            return torch.zeros_like(xt)

    loaded_prior = types.SimpleNamespace(
        model=_DummyModel(),
        scheduler=_DummyScheduler(),
        style_id=None,
        window_size=4,
        feature_dim=12,
        num_diffusion_steps=10,
    )
    monkeypatch.setattr(module, "load_smp_prior_checkpoint", lambda *args, **kwargs: loaded_prior)

    result = module.prepare_smp_denoising_diagnostic_clip(
        dataset=dataset_path,
        checkpoint=tmp_path / "dummy_checkpoint.pt",
        clip_start_sec=0.0,
        clip_duration_sec=0.12,
        noise_step=3,
        stride=2,
        noise=torch.full((2, 4, 12), 0.1, dtype=torch.float32),
    )

    assert tuple(result.clean_frames.shape) == (6, 12)
    assert tuple(result.noisy_frames.shape) == (6, 12)
    assert tuple(result.denoised_frames.shape) == (6, 12)
    assert set(result.decoded_state_map) == {"clean_decoded", "noisy_decoded", "denoised_decoded"}
    assert result.raw_joint_names == ("raw_joint_a", "raw_joint_b")
    assert result.decoded_joint_names == ("smp_joint",)
    assert "clean_vs_noisy_mse" in result.metrics
    assert "clean_vs_denoised_mse" in result.metrics
    assert "clean_decoded_vs_raw_joint_mae" in result.metrics
    assert result.metrics["clean_decoded_vs_raw_joint_mae"] == 0.0


def test_prepare_diagnostic_clip_reuses_raw_root_carrier_for_decoded_states(tmp_path, monkeypatch):
    module = _load_smp_denoising_diagnostic_module()
    dataset_path = _make_toy_paired_dataset_with_moving_raw_root(tmp_path / "dataset_walk_root.npz")

    class _DummyScheduler:
        def __init__(self):
            self.alpha_bar = torch.linspace(1.0, 0.1, 10, dtype=torch.float32)

        def q_sample(self, windows, timesteps, noise):
            del timesteps
            return windows + 0.25 * noise

    class _DummyModel:
        num_styles = 0

        def __call__(self, xt, timesteps, style_id=None):
            del timesteps, style_id
            return torch.zeros_like(xt)

    loaded_prior = types.SimpleNamespace(
        model=_DummyModel(),
        scheduler=_DummyScheduler(),
        style_id=None,
        window_size=4,
        feature_dim=12,
        num_diffusion_steps=10,
    )
    monkeypatch.setattr(module, "load_smp_prior_checkpoint", lambda *args, **kwargs: loaded_prior)

    result = module.prepare_smp_denoising_diagnostic_clip(
        dataset=dataset_path,
        checkpoint=tmp_path / "dummy_checkpoint.pt",
        clip_start_sec=0.0,
        clip_duration_sec=0.12,
        noise_step=3,
        stride=2,
        noise=torch.full((2, 4, 12), 0.1, dtype=torch.float32),
    )

    clean_decoded = result.decoded_state_map["clean_decoded"]
    assert torch.equal(clean_decoded.root_pos_w, result.raw_clip.root_pos_w)
    assert torch.equal(clean_decoded.root_quat_w, result.raw_clip.root_quat_w)
