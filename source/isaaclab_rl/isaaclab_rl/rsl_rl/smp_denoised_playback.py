# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from rsl_rl.diffusion.model import MotionEpsilonTransformer
    from rsl_rl.diffusion.scheduler import DiffusionScheduler


@dataclass
class PlaybackStateSequence:
    """Approximate playback states with yaw-only heading and planar root translation.

    Returned tensors are batched per frame:
    - `root_pos_w`: `(T, 3)` world root position with fixed height from reference root.
    - `root_quat_w`: `(T, 4)` world root orientation reconstructed from integrated heading yaw.
    - `root_lin_vel_w`: `(T, 3)` world root linear velocity using planar (height-fixed) approximation.
    - `root_ang_vel_w`: `(T, 3)` world root angular velocity from heading-local angular features.
    - `joint_pos`: `(T, J)` joint positions from decoded rot6d offsets + default joint pose.
    - `joint_vel`: `(T, J)` finite-difference joint velocities.
    """

    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass
class LoadedSmpPrior:
    model: MotionEpsilonTransformer
    scheduler: DiffusionScheduler
    style_id: int | None
    window_size: int
    feature_dim: int
    num_diffusion_steps: int


def _normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected rot6d last dim 6, got {rot6d.shape[-1]}")

    col_1 = torch.stack((rot6d[..., 0], rot6d[..., 2], rot6d[..., 4]), dim=-1)
    col_2 = torch.stack((rot6d[..., 1], rot6d[..., 3], rot6d[..., 5]), dim=-1)
    basis_1 = _normalize(col_1)
    basis_2 = _normalize(col_2 - (basis_1 * col_2).sum(dim=-1, keepdim=True) * basis_1)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def _joint_rot6d_to_angle_offsets(joint_rot6d: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    rotmat = _rot6d_to_matrix(joint_rot6d)
    normalized_axes = _normalize(joint_axes.to(device=joint_rot6d.device, dtype=joint_rot6d.dtype))
    axis_shape = (1,) * (joint_rot6d.ndim - 2) + normalized_axes.shape
    expanded_axes = normalized_axes.view(axis_shape)

    cos_theta = ((torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew_vec = torch.stack(
        (
            rotmat[..., 2, 1] - rotmat[..., 1, 2],
            rotmat[..., 0, 2] - rotmat[..., 2, 0],
            rotmat[..., 1, 0] - rotmat[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = 0.5 * (skew_vec * expanded_axes).sum(dim=-1)
    return torch.atan2(sin_theta, cos_theta)


def _heading_local_to_world(yaw: torch.Tensor, vec_local: torch.Tensor) -> torch.Tensor:
    """Convert heading-local vectors (forward, up, lateral) to world XYZ."""
    if vec_local.shape[-1] != 3:
        raise ValueError(f"Expected vec_local last dim 3, got {vec_local.shape[-1]}")
    if yaw.ndim != 1 or yaw.shape[0] != vec_local.shape[0]:
        raise ValueError(f"Expected yaw shape ({vec_local.shape[0]},), got {yaw.shape}")

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    forward = vec_local[:, 0]
    up = vec_local[:, 1]
    lateral = vec_local[:, 2]
    return torch.stack(
        (
            cos_yaw * forward + sin_yaw * lateral,
            sin_yaw * forward - cos_yaw * lateral,
            up,
        ),
        dim=-1,
    )


def decode_joint_positions_from_rot6d(
    joint_rot6d_rel: torch.Tensor,
    joint_axes: torch.Tensor,
    default_joint_pos: torch.Tensor,
) -> torch.Tensor:
    if joint_rot6d_rel.ndim < 2:
        raise ValueError(f"Expected joint_rot6d_rel with shape (..., num_joints, 6), got {joint_rot6d_rel.shape}")
    if joint_axes.ndim != 2 or joint_axes.shape[-1] != 3:
        raise ValueError(f"Expected joint_axes shape (num_joints, 3), got {joint_axes.shape}")
    if joint_rot6d_rel.shape[-1] != 6:
        raise ValueError(f"Expected joint_rot6d_rel last dim 6, got {joint_rot6d_rel.shape[-1]}")
    if joint_rot6d_rel.shape[-2] != joint_axes.shape[0]:
        raise ValueError(
            f"Expected joint_rot6d_rel joint dim {joint_axes.shape[0]}, got {joint_rot6d_rel.shape[-2]}"
        )

    joint_offsets = _joint_rot6d_to_angle_offsets(joint_rot6d_rel, joint_axes)
    default = default_joint_pos.to(device=joint_offsets.device, dtype=joint_offsets.dtype)
    if default.ndim != 1 or default.shape[0] != joint_axes.shape[0]:
        raise ValueError(f"Expected default_joint_pos shape ({joint_axes.shape[0]},), got {default_joint_pos.shape}")

    default = default.view(*([1] * (joint_offsets.ndim - 1)), -1)
    return default + joint_offsets


def reconstruct_playback_states(
    denoised_frame_features: torch.Tensor,
    *,
    feature_block_offsets: dict[str, tuple[int, int]],
    joint_axes: torch.Tensor,
    default_joint_pos: torch.Tensor,
    reference_root_pos_w: torch.Tensor,
    dt: float,
    initial_heading_yaw: float = 0.0,
) -> PlaybackStateSequence:
    if denoised_frame_features.ndim != 2:
        raise ValueError(f"Expected denoised_frame_features shape (num_frames, feature_dim), got {denoised_frame_features.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    required = ("base_lin_vel_b", "base_ang_vel_b", "joint_rot6d_rel")
    missing = [key for key in required if key not in feature_block_offsets]
    if missing:
        raise KeyError(f"Missing required feature block offsets: {missing}")

    num_frames = int(denoised_frame_features.shape[0])
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    device = denoised_frame_features.device
    dtype = denoised_frame_features.dtype

    lin_start, lin_stop = feature_block_offsets["base_lin_vel_b"]
    ang_start, ang_stop = feature_block_offsets["base_ang_vel_b"]
    joint_start, joint_stop = feature_block_offsets["joint_rot6d_rel"]

    base_lin_vel_b = denoised_frame_features[:, lin_start:lin_stop]
    base_ang_vel_b = denoised_frame_features[:, ang_start:ang_stop]
    joint_rot6d_rel = denoised_frame_features[:, joint_start:joint_stop]

    if base_lin_vel_b.shape[-1] != 3 or base_ang_vel_b.shape[-1] != 3:
        raise ValueError("base_lin_vel_b and base_ang_vel_b blocks must both have width 3")
    if joint_rot6d_rel.shape[-1] % 6 != 0:
        raise ValueError("joint_rot6d_rel block width must be divisible by 6")

    num_joints = joint_rot6d_rel.shape[-1] // 6
    joint_rot6d_rel = joint_rot6d_rel.reshape(num_frames, num_joints, 6)
    joint_pos = decode_joint_positions_from_rot6d(
        joint_rot6d_rel=joint_rot6d_rel,
        joint_axes=joint_axes,
        default_joint_pos=default_joint_pos,
    )

    yaw_rate = base_ang_vel_b[:, 1]
    yaw = torch.zeros((num_frames,), device=device, dtype=dtype)
    yaw[0] = float(initial_heading_yaw)
    if num_frames > 1:
        yaw[1:] = yaw[0] + torch.cumsum(yaw_rate[:-1] * dt, dim=0)

    base_lin_vel_b_planar = base_lin_vel_b.clone()
    base_lin_vel_b_planar[:, 1] = 0.0
    root_lin_vel_w = _heading_local_to_world(yaw, base_lin_vel_b_planar)
    root_ang_vel_w = _heading_local_to_world(yaw, base_ang_vel_b)

    reference_root = reference_root_pos_w.to(device=device, dtype=dtype)
    if reference_root.shape != (3,):
        raise ValueError(f"Expected reference_root_pos_w shape (3,), got {reference_root_pos_w.shape}")

    root_pos_w = reference_root.unsqueeze(0).repeat(num_frames, 1)
    if num_frames > 1:
        root_pos_w[1:, :2] = reference_root[:2] + torch.cumsum(root_lin_vel_w[:-1, :2] * dt, dim=0)
    root_pos_w[:, 2] = reference_root[2]

    root_quat_w = torch.zeros((num_frames, 4), device=device, dtype=dtype)
    root_quat_w[:, 0] = torch.cos(0.5 * yaw)
    root_quat_w[:, 3] = torch.sin(0.5 * yaw)

    joint_vel = torch.zeros_like(joint_pos)
    if num_frames > 1:
        joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt
        joint_vel[0] = joint_vel[1]

    return PlaybackStateSequence(
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        root_lin_vel_w=root_lin_vel_w,
        root_ang_vel_w=root_ang_vel_w,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )


def resolve_dataset_npz(dataset_path: str | Path, source_name: str | None = None) -> Path:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected .npz file, got: {path}")
        return path

    candidates = sorted(path.glob("dataset_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No dataset_*.npz files found in directory: {path}")

    if source_name is None:
        return candidates[0]

    for candidate in candidates:
        with np.load(candidate) as data:
            if "source_name" not in data:
                continue
            source_values = np.asarray(data["source_name"]).reshape(-1)
            if source_values.size == 0:
                raise ValueError(f"Malformed source_name metadata in dataset file: {candidate}")
            candidate_source = str(source_values[0])
            if candidate_source.strip() == "":
                raise ValueError(f"Malformed source_name metadata in dataset file: {candidate}")
        if candidate_source == source_name:
            return candidate

    raise FileNotFoundError(f"Could not find dataset npz for source_name={source_name!r} in: {path}")


def slice_clip_frames(
    frames: torch.Tensor,
    fps: float,
    clip_start_sec: float,
    clip_duration_sec: float,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Slice clip frames using truncation-based seconds-to-frames conversion: int(seconds * fps)."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if clip_start_sec < 0.0:
        raise ValueError(f"clip_start_sec must be non-negative, got {clip_start_sec}")
    if clip_duration_sec <= 0.0:
        raise ValueError(f"clip_duration_sec must be positive, got {clip_duration_sec}")

    start_idx = int(clip_start_sec * fps)
    duration_frames = int(clip_duration_sec * fps)
    if duration_frames <= 0:
        raise ValueError("Requested clip duration resolves to zero frames")
    end_idx = start_idx + duration_frames

    if start_idx < 0 or end_idx > int(frames.shape[0]):
        raise ValueError(
            f"Requested clip [{start_idx}, {end_idx}) is out of range for total_frames={int(frames.shape[0])}"
        )

    return frames[start_idx:end_idx], (start_idx, end_idx)


def build_motion_windows(frames: torch.Tensor, window_size: int, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice `(T, F...)` frames into overlapping windows and their start indices."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if frames.shape[0] < window_size:
        raise ValueError(f"window_size={window_size} exceeds total frames={int(frames.shape[0])}")

    last_start = int(frames.shape[0]) - window_size
    starts = torch.arange(
        0,
        last_start + 1,
        stride,
        device=frames.device,
        dtype=torch.long,
    )
    if int(starts[-1]) != last_start:
        starts = torch.cat([starts, torch.tensor([last_start], device=frames.device, dtype=torch.long)])

    windows = torch.stack([frames[start : start + window_size] for start in starts.tolist()], dim=0)
    return windows, starts


def reconstruct_x0_from_eps(xt: torch.Tensor, eps_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    """Reconstruct denoised `x0_hat` from `(x_t, eps_hat, alpha_bar_t)` with DDPM closed form."""
    if xt.shape != eps_hat.shape:
        raise ValueError(f"xt and eps_hat must share shape, got {xt.shape} and {eps_hat.shape}")

    alpha_bar = alpha_bar_t.reshape(-1).to(device=xt.device, dtype=xt.dtype)
    batch_size = int(xt.shape[0])
    if alpha_bar.numel() == 1:
        alpha_bar = alpha_bar.expand(batch_size)
    elif alpha_bar.numel() != batch_size:
        raise ValueError(f"alpha_bar_t must have 1 or {batch_size} elements, got {alpha_bar.numel()}")

    alpha_bar = alpha_bar.view(-1, *([1] * (xt.ndim - 1)))
    return (xt - (1.0 - alpha_bar).sqrt() * eps_hat) / alpha_bar.sqrt()


def denoise_motion_windows(
    windows: torch.Tensor,
    model,
    scheduler,
    timesteps: torch.Tensor,
    style_id: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward-noise windows with `scheduler.q_sample` and reconstruct `x0_hat` from model-predicted epsilon."""
    if timesteps.ndim != 1 or timesteps.shape[0] != windows.shape[0]:
        raise ValueError(f"timesteps must have shape ({windows.shape[0]},), got {timesteps.shape}")

    eps = torch.randn_like(windows) if noise is None else noise
    xt = scheduler.q_sample(windows, timesteps, eps)

    if style_id is None:
        eps_hat = model(xt, timesteps)
    else:
        eps_hat = model(xt, timesteps, style_id=style_id)

    alpha_bar_t = scheduler.alpha_bar.to(device=windows.device, dtype=windows.dtype)[timesteps]
    return reconstruct_x0_from_eps(xt, eps_hat, alpha_bar_t)


def stitch_motion_windows(windows: torch.Tensor, starts: torch.Tensor, total_frames: int) -> torch.Tensor:
    """Average overlapping window predictions back into a continuous `(total_frames, F...)` clip."""
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if windows.ndim < 2:
        raise ValueError(f"windows must have shape (num_windows, window_size, ...), got {windows.shape}")
    if starts.ndim != 1 or starts.shape[0] != windows.shape[0]:
        raise ValueError(f"starts must have shape ({windows.shape[0]},), got {starts.shape}")

    window_size = int(windows.shape[1])
    feature_shape = tuple(windows.shape[2:])
    stitched = torch.zeros((total_frames, *feature_shape), dtype=windows.dtype, device=windows.device)
    counts = torch.zeros((total_frames,), dtype=windows.dtype, device=windows.device)

    for window_idx, start in enumerate(starts.tolist()):
        end = start + window_size
        if start < 0 or end > total_frames:
            raise ValueError(f"Window range [{start}, {end}) out of bounds for total_frames={total_frames}")
        stitched[start:end] += windows[window_idx]
        counts[start:end] += 1.0

    if torch.any(counts == 0):
        raise ValueError("Window coverage has gaps; cannot stitch all frames")

    denom = counts.view(-1, *([1] * len(feature_shape)))
    return stitched / denom


def _find_nested_rsl_rl_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "rsl_rl" / "rsl_rl" / "__init__.py"
        if candidate.exists():
            return parent / "rsl_rl"
    raise FileNotFoundError("Could not locate nested rsl_rl repository")


def _clear_shadowed_rsl_rl_module(rsl_rl_repo_root: Path) -> None:
    existing_module = sys.modules.get("rsl_rl")
    expected_init = (rsl_rl_repo_root / "rsl_rl" / "__init__.py").resolve()
    if existing_module is None:
        return

    module_file = getattr(existing_module, "__file__", None)
    if module_file is not None and Path(module_file).resolve() == expected_init:
        return

    for module_name in list(sys.modules):
        if module_name == "rsl_rl" or module_name.startswith("rsl_rl."):
            del sys.modules[module_name]
    importlib.invalidate_caches()


def _import_diffusion_runtime():
    try:
        from rsl_rl.diffusion.model import MotionEpsilonTransformer
        from rsl_rl.diffusion.scheduler import DiffusionScheduler
    except ModuleNotFoundError:
        rsl_rl_repo_root = _find_nested_rsl_rl_repo_root()
        if str(rsl_rl_repo_root) not in sys.path:
            sys.path.insert(0, str(rsl_rl_repo_root))
        _clear_shadowed_rsl_rl_module(rsl_rl_repo_root)
        from rsl_rl.diffusion.model import MotionEpsilonTransformer
        from rsl_rl.diffusion.scheduler import DiffusionScheduler

    return MotionEpsilonTransformer, DiffusionScheduler


def _extract_scalar_from_npz_field(value: object, field_name: str) -> float:
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Dataset metadata field {field_name!r} is empty")
    return float(values[0])


def load_smp_prior_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    style_id: int | None = None,
) -> LoadedSmpPrior:
    resolved_checkpoint = Path(checkpoint_path)
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {resolved_checkpoint}")

    runtime_device = torch.device(device) if device is not None else torch.device("cpu")
    checkpoint = torch.load(resolved_checkpoint, map_location=runtime_device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(checkpoint).__name__}")
    required_keys = ("model_state_dict", "model_cfg")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(f"Checkpoint is missing required keys: {missing_keys}")

    model_cfg = checkpoint["model_cfg"]
    if not isinstance(model_cfg, Mapping):
        raise TypeError(f"Expected checkpoint model_cfg to be a mapping, got {type(model_cfg).__name__}")

    required_model_cfg_keys = ("feature_dim", "window_size", "num_diffusion_steps")
    missing_cfg = [key for key in required_model_cfg_keys if key not in model_cfg]
    if missing_cfg:
        raise KeyError(f"Checkpoint model_cfg is missing required keys: {missing_cfg}")
    model_state_dict = checkpoint["model_state_dict"]
    if not isinstance(model_state_dict, Mapping):
        raise TypeError(
            f"Expected checkpoint model_state_dict to be a mapping, got {type(model_state_dict).__name__}"
        )

    MotionEpsilonTransformer, DiffusionScheduler = _import_diffusion_runtime()
    model = MotionEpsilonTransformer(**model_cfg).to(runtime_device)
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    num_diffusion_steps = int(model_cfg["num_diffusion_steps"])
    scheduler_cfg = checkpoint.get("scheduler_cfg")
    scheduler_kwargs: dict[str, float] = {}
    if scheduler_cfg is not None:
        if not isinstance(scheduler_cfg, Mapping):
            raise TypeError(f"Expected scheduler_cfg to be a mapping, got {type(scheduler_cfg).__name__}")
        if "beta_start" in scheduler_cfg:
            scheduler_kwargs["beta_start"] = float(scheduler_cfg["beta_start"])
        if "beta_end" in scheduler_cfg:
            scheduler_kwargs["beta_end"] = float(scheduler_cfg["beta_end"])
    scheduler = DiffusionScheduler(num_steps=num_diffusion_steps, **scheduler_kwargs)

    return LoadedSmpPrior(
        model=model,
        scheduler=scheduler,
        style_id=style_id,
        window_size=int(model_cfg["window_size"]),
        feature_dim=int(model_cfg["feature_dim"]),
        num_diffusion_steps=num_diffusion_steps,
    )
