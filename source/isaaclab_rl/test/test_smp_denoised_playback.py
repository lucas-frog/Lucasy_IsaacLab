# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import importlib.machinery
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_smp_denoised_playback_module():
    # Locate the source file by traversing parent directories to avoid fixed cwd assumptions.
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "isaaclab_rl" / "rsl_rl" / "smp_denoised_playback.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_denoised_playback_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find isaaclab_rl/rsl_rl/smp_denoised_playback.py")


def _load_diffusion_scheduler_module():
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "rsl_rl" / "rsl_rl" / "diffusion" / "scheduler.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_scheduler_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find rsl_rl/rsl_rl/diffusion/scheduler.py")


def _load_diffusion_model_module():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "rsl_rl" / "rsl_rl" / "__init__.py"
        if candidate.exists():
            repo_root = parent / "rsl_rl"
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            break

    for parent in Path(__file__).resolve().parents:
        module_path = parent / "rsl_rl" / "rsl_rl" / "diffusion" / "model.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_model_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find rsl_rl/rsl_rl/diffusion/model.py")


def _quat_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat_wxyz.shape[:-1], 3, 3)


def _joint_offsets_to_rot6d(offsets: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    axes = joint_axes.to(device=offsets.device, dtype=offsets.dtype)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    expanded_axes = axes.view(*([1] * (offsets.ndim - 1)), *axes.shape)
    half_angle = 0.5 * offsets
    quat = torch.cat(
        (
            torch.cos(half_angle).unsqueeze(-1),
            expanded_axes * torch.sin(half_angle).unsqueeze(-1),
        ),
        dim=-1,
    )
    return _quat_to_matrix(quat)[..., :2].reshape(*quat.shape[:-1], 6)


def test_resolve_dataset_npz_from_single_file(tmp_path):
    module = _load_smp_denoised_playback_module()
    dataset_path = tmp_path / "dataset_walk_a.npz"
    np.savez(dataset_path, frames=np.zeros((8, 4), dtype=np.float32), fps=np.array([50.0], dtype=np.float32))

    resolved = module.resolve_dataset_npz(dataset_path)

    assert resolved == dataset_path


def test_resolve_dataset_npz_from_corpus_dir_by_source_name(tmp_path):
    module = _load_smp_denoised_playback_module()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    (corpus_dir / "style_vocab.json").write_text("{}", encoding="utf-8")
    walk_path = corpus_dir / "dataset_walk_clip.npz"
    run_path = corpus_dir / "dataset_run_clip.npz"
    np.savez(
        walk_path,
        frames=np.zeros((8, 4), dtype=np.float32),
        fps=np.array([50.0], dtype=np.float32),
        source_name=np.asarray(["walk1"], dtype=np.str_),
    )
    np.savez(
        run_path,
        frames=np.zeros((8, 4), dtype=np.float32),
        fps=np.array([50.0], dtype=np.float32),
        source_name=np.asarray(["run1"], dtype=np.str_),
    )

    resolved = module.resolve_dataset_npz(corpus_dir, source_name="run1")

    assert resolved == run_path


def test_resolve_dataset_npz_from_corpus_dir_defaults_to_first_dataset_when_source_name_missing(tmp_path):
    module = _load_smp_denoised_playback_module()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    first_path = corpus_dir / "dataset_a_clip.npz"
    second_path = corpus_dir / "dataset_b_clip.npz"
    np.savez(
        first_path,
        frames=np.zeros((4, 2), dtype=np.float32),
        fps=np.array([50.0], dtype=np.float32),
        source_name=np.asarray(["clip_a"], dtype=np.str_),
    )
    np.savez(
        second_path,
        frames=np.zeros((4, 2), dtype=np.float32),
        fps=np.array([50.0], dtype=np.float32),
        source_name=np.asarray(["clip_b"], dtype=np.str_),
    )

    resolved = module.resolve_dataset_npz(corpus_dir)

    assert resolved == first_path


def test_resolve_dataset_npz_rejects_empty_source_name_metadata(tmp_path):
    module = _load_smp_denoised_playback_module()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    broken_path = corpus_dir / "dataset_broken.npz"
    np.savez(
        broken_path,
        frames=np.zeros((4, 2), dtype=np.float32),
        fps=np.array([50.0], dtype=np.float32),
        source_name=np.asarray([], dtype=np.str_),
    )

    with pytest.raises(ValueError, match="source_name"):
        module.resolve_dataset_npz(corpus_dir, source_name="target_clip")


def test_resolve_dataset_npz_rejects_blank_source_name_metadata(tmp_path):
    module = _load_smp_denoised_playback_module()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    broken_path = corpus_dir / "dataset_broken_blank.npz"
    np.savez(
        broken_path,
        frames=np.zeros((4, 2), dtype=np.float32),
        fps=np.array([50.0], dtype=np.float32),
        source_name=np.asarray(["   \n"], dtype=np.str_),
    )

    with pytest.raises(ValueError, match="source_name"):
        module.resolve_dataset_npz(corpus_dir, source_name="target_clip")


def test_slice_clip_frames_uses_fps_and_duration():
    module = _load_smp_denoised_playback_module()
    frames = torch.arange(300, dtype=torch.float32).reshape(100, 3)

    clip, frame_range = module.slice_clip_frames(frames, fps=50.0, clip_start_sec=1.0, clip_duration_sec=0.5)

    assert frame_range == (50, 75)
    assert torch.equal(clip, frames[50:75])


def test_slice_clip_frames_truncates_fractional_seconds():
    module = _load_smp_denoised_playback_module()
    frames = torch.arange(300, dtype=torch.float32).reshape(100, 3)

    clip, frame_range = module.slice_clip_frames(frames, fps=10.0, clip_start_sec=0.19, clip_duration_sec=0.21)

    assert frame_range == (1, 3)
    assert torch.equal(clip, frames[1:3])


def test_slice_clip_frames_rejects_out_of_range_request():
    module = _load_smp_denoised_playback_module()
    frames = torch.randn(100, 4)

    with pytest.raises(ValueError):
        module.slice_clip_frames(frames, fps=50.0, clip_start_sec=1.9, clip_duration_sec=0.2)


def test_build_motion_windows_returns_expected_shape():
    module = _load_smp_denoised_playback_module()
    frames = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    windows, starts = module.build_motion_windows(frames, window_size=4, stride=2)

    assert windows.shape == (2, 4, 2)
    assert torch.equal(starts, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(windows[0], frames[0:4])
    assert torch.equal(windows[1], frames[2:6])


def test_build_motion_windows_appends_tail_start_for_full_coverage():
    module = _load_smp_denoised_playback_module()
    frames = torch.arange(10, dtype=torch.float32).reshape(10, 1)

    windows, starts = module.build_motion_windows(frames, window_size=4, stride=4)

    assert torch.equal(starts, torch.tensor([0, 4, 6], dtype=torch.long))
    stitched = module.stitch_motion_windows(windows, starts, total_frames=int(frames.shape[0]))
    assert torch.equal(stitched, frames)


def test_stitch_motion_windows_averages_overlaps():
    module = _load_smp_denoised_playback_module()
    windows = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [22.0]],
        ],
        dtype=torch.float32,
    )
    starts = torch.tensor([0, 1, 2], dtype=torch.long)

    stitched = module.stitch_motion_windows(windows, starts, total_frames=5)

    expected = torch.tensor([[0.0], [5.5], [11.0], [16.5], [22.0]], dtype=torch.float32)
    assert torch.allclose(stitched, expected)


def test_reconstruct_x0_from_eps_matches_closed_form_solution():
    module = _load_smp_denoised_playback_module()
    x0 = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]], dtype=torch.float32)
    eps = torch.tensor([[[0.2, -0.4], [0.1, 0.3]]], dtype=torch.float32)
    alpha_bar_t = torch.tensor([0.64], dtype=torch.float32)
    xt = alpha_bar_t.view(1, 1, 1).sqrt() * x0 + (1.0 - alpha_bar_t.view(1, 1, 1)).sqrt() * eps

    x0_hat = module.reconstruct_x0_from_eps(xt, eps, alpha_bar_t)

    assert torch.allclose(x0_hat, x0)


def test_denoise_motion_windows_uses_model_style_id_when_available():
    module = _load_smp_denoised_playback_module()
    scheduler_module = _load_diffusion_scheduler_module()
    scheduler = scheduler_module.DiffusionScheduler(num_steps=10)
    windows = torch.randn(2, 4, 3)
    timesteps = torch.tensor([1, 3], dtype=torch.long)
    style_id = torch.tensor([2, 5], dtype=torch.long)

    class _RecordingModel:
        def __init__(self):
            self.last_style_id = None

        def __call__(self, xt, t, style_id=None):
            self.last_style_id = style_id
            return torch.zeros_like(xt)

    model = _RecordingModel()
    called = {"count": 0}
    original_q_sample = scheduler.q_sample

    def _recording_q_sample(x0, t, eps):
        called["count"] += 1
        return original_q_sample(x0, t, eps)

    scheduler.q_sample = _recording_q_sample
    output = module.denoise_motion_windows(
        windows,
        model=model,
        scheduler=scheduler,
        timesteps=timesteps,
        style_id=style_id,
        noise=torch.zeros_like(windows),
    )

    assert output.shape == windows.shape
    assert called["count"] == 1
    assert torch.equal(model.last_style_id, style_id)


def test_denoise_motion_windows_matches_closed_form_without_style_id():
    module = _load_smp_denoised_playback_module()
    scheduler_module = _load_diffusion_scheduler_module()
    scheduler = scheduler_module.DiffusionScheduler(num_steps=10)
    windows = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ],
        dtype=torch.float32,
    )
    noise = torch.tensor(
        [
            [[0.1], [0.2]],
            [[0.3], [0.4]],
        ],
        dtype=torch.float32,
    )
    timesteps = torch.tensor([2, 5], dtype=torch.long)

    class _TwoArgModel:
        def __init__(self):
            self.calls = 0
            self.last_timesteps = None

        def __call__(self, xt, t):
            self.calls += 1
            self.last_timesteps = t
            return torch.full_like(xt, 0.25)

    model = _TwoArgModel()
    output = module.denoise_motion_windows(
        windows,
        model=model,
        scheduler=scheduler,
        timesteps=timesteps,
        noise=noise,
    )

    xt = scheduler.q_sample(windows, timesteps, noise)
    alpha_bar = scheduler.alpha_bar[timesteps].view(-1, 1, 1)
    eps_hat = torch.full_like(xt, 0.25)
    expected = (xt - (1.0 - alpha_bar).sqrt() * eps_hat) / alpha_bar.sqrt()
    assert torch.allclose(output, expected)
    assert model.calls == 1
    assert torch.equal(model.last_timesteps, timesteps)


def test_decode_joint_positions_from_rot6d_recovers_default_pose_offsets():
    module = _load_smp_denoised_playback_module()
    joint_axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    default_joint_pos = torch.tensor([0.2, -0.4], dtype=torch.float32)
    joint_offsets = torch.tensor(
        [
            [0.0, 0.1],
            [0.3, -0.2],
            [-0.5, 0.4],
        ],
        dtype=torch.float32,
    )
    joint_rot6d_rel = _joint_offsets_to_rot6d(joint_offsets, joint_axes)

    joint_pos = module.decode_joint_positions_from_rot6d(
        joint_rot6d_rel=joint_rot6d_rel,
        joint_axes=joint_axes,
        default_joint_pos=default_joint_pos,
    )

    assert torch.allclose(joint_pos, default_joint_pos + joint_offsets, atol=1.0e-5)


def test_reconstruct_playback_states_integrates_heading_and_position():
    module = _load_smp_denoised_playback_module()
    feature_block_offsets = {
        "base_lin_vel_b": (0, 3),
        "base_ang_vel_b": (3, 6),
        "joint_rot6d_rel": (6, 12),
    }
    joint_axes = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    default_joint_pos = torch.tensor([0.5], dtype=torch.float32)
    joint_offsets = torch.tensor([[0.0], [0.1], [0.3], [0.6]], dtype=torch.float32)
    frames = torch.cat(
        (
            torch.tensor(
                [
                    [1.0, 0.25, 0.0],
                    [1.0, 0.25, 0.0],
                    [1.0, 0.25, 0.0],
                    [1.0, 0.25, 0.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, math.pi / 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            _joint_offsets_to_rot6d(joint_offsets, joint_axes).reshape(4, -1),
        ),
        dim=-1,
    )

    states = module.reconstruct_playback_states(
        denoised_frame_features=frames,
        feature_block_offsets=feature_block_offsets,
        joint_axes=joint_axes,
        default_joint_pos=default_joint_pos,
        reference_root_pos_w=torch.tensor([0.0, 0.0, 0.75], dtype=torch.float32),
        dt=1.0,
    )

    assert isinstance(states, module.PlaybackStateSequence)
    assert torch.allclose(
        states.root_pos_w,
        torch.tensor(
            [
                [0.0, 0.0, 0.75],
                [1.0, 0.0, 0.75],
                [2.0, 0.0, 0.75],
                [2.0, 1.0, 0.75],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-5,
    )
    assert torch.allclose(
        states.root_quat_w,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
                [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-5,
    )
    assert torch.allclose(
        states.root_lin_vel_w,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-5,
    )
    assert torch.allclose(
        states.root_ang_vel_w,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, math.pi / 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-5,
    )
    assert torch.allclose(states.joint_pos.squeeze(-1), torch.tensor([0.5, 0.6, 0.8, 1.1], dtype=torch.float32))
    assert torch.allclose(
        states.joint_vel[1:].squeeze(-1),
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        atol=1.0e-5,
    )


def test_reconstruct_playback_states_rejects_empty_sequence():
    module = _load_smp_denoised_playback_module()

    with pytest.raises(ValueError, match="num_frames must be positive"):
        module.reconstruct_playback_states(
            denoised_frame_features=torch.zeros((0, 12), dtype=torch.float32),
            feature_block_offsets={
                "base_lin_vel_b": (0, 3),
                "base_ang_vel_b": (3, 6),
                "joint_rot6d_rel": (6, 12),
            },
            joint_axes=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            default_joint_pos=torch.tensor([0.0], dtype=torch.float32),
            reference_root_pos_w=torch.tensor([0.0, 0.0, 0.75], dtype=torch.float32),
            dt=1.0,
        )


def test_load_smp_prior_checkpoint_restores_model_cfg_and_style_id(tmp_path):
    module = _load_smp_denoised_playback_module()
    model_module = _load_diffusion_model_module()
    model_cfg = {
        "feature_dim": 12,
        "window_size": 4,
        "num_diffusion_steps": 10,
        "num_styles": 3,
        "hidden_dim": 24,
        "num_layers": 1,
        "num_heads": 4,
    }
    model = model_module.MotionEpsilonTransformer(**model_cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.token_proj.weight.fill_(0.25)

    checkpoint_path = tmp_path / "model_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "style_cfg": {"style_names": ["walk", "run", "jump"], "style_to_id": {"walk": 0, "run": 1, "jump": 2}},
        },
        checkpoint_path,
    )

    loaded = module.load_smp_prior_checkpoint(checkpoint_path, device="cpu", style_id=2)

    assert loaded.feature_dim == model_cfg["feature_dim"]
    assert loaded.window_size == model_cfg["window_size"]
    assert loaded.num_diffusion_steps == model_cfg["num_diffusion_steps"]
    assert loaded.style_id == 2
    assert loaded.scheduler.num_steps == model_cfg["num_diffusion_steps"]
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    assert torch.allclose(loaded.model.token_proj.weight, model.token_proj.weight)


def test_import_diffusion_runtime_recovers_from_outer_namespace_package(monkeypatch):
    module = _load_smp_denoised_playback_module()
    outer_repo = next(parent / "rsl_rl" for parent in Path(__file__).resolve().parents if (parent / "rsl_rl" / "rsl_rl" / "__init__.py").exists())
    namespace_pkg = types.ModuleType("rsl_rl")
    namespace_pkg.__path__ = [str(outer_repo)]
    namespace_pkg.__package__ = "rsl_rl"
    namespace_pkg.__spec__ = importlib.machinery.ModuleSpec("rsl_rl", loader=None, is_package=True)
    namespace_pkg.__spec__.submodule_search_locations = [str(outer_repo)]
    for module_name in list(sys.modules):
        if module_name == "rsl_rl" or module_name.startswith("rsl_rl."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "rsl_rl", namespace_pkg)

    MotionEpsilonTransformer, DiffusionScheduler = module._import_diffusion_runtime()

    assert MotionEpsilonTransformer.__name__ == "MotionEpsilonTransformer"
    assert DiffusionScheduler.__name__ == "DiffusionScheduler"
    resolved_rsl_rl = sys.modules["rsl_rl"]
    module_file = getattr(resolved_rsl_rl, "__file__", None)
    if module_file is not None:
        assert Path(module_file).resolve() == outer_repo / "rsl_rl" / "__init__.py"
    else:
        assert str(outer_repo) in list(getattr(resolved_rsl_rl, "__path__", []))
    assert "rsl_rl.diffusion.model" in sys.modules


def test_load_smp_prior_checkpoint_restores_scheduler_cfg(tmp_path):
    module = _load_smp_denoised_playback_module()
    model_module = _load_diffusion_model_module()
    model_cfg = {
        "feature_dim": 12,
        "window_size": 4,
        "num_diffusion_steps": 8,
        "num_styles": 0,
        "hidden_dim": 24,
        "num_layers": 1,
        "num_heads": 4,
    }
    model = model_module.MotionEpsilonTransformer(**model_cfg)

    checkpoint_path = tmp_path / "model_with_scheduler_cfg.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "scheduler_cfg": {"beta_start": 5.0e-4, "beta_end": 5.0e-2},
        },
        checkpoint_path,
    )

    loaded = module.load_smp_prior_checkpoint(checkpoint_path, device="cpu")

    assert loaded.scheduler.num_steps == model_cfg["num_diffusion_steps"]
    assert loaded.scheduler.beta[0].item() == pytest.approx(5.0e-4, rel=1.0e-6, abs=1.0e-7)
    assert loaded.scheduler.beta[-1].item() == pytest.approx(5.0e-2, rel=1.0e-6, abs=1.0e-7)

