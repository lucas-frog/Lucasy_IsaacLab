# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .smp_denoised_playback import (
    PlaybackStateSequence,
    build_motion_windows,
    denoise_motion_windows,
    load_smp_prior_checkpoint,
    reconstruct_playback_states,
    resolve_dataset_npz,
    slice_clip_frames,
    stitch_motion_windows,
)
from .smp_paired_dataset import (
    PairedRawStateClip,
    build_g1_default_joint_pos,
    load_paired_raw_state_clip,
    load_smp_paired_dataset,
    read_feature_block_offsets,
    read_replay_joint_names,
    should_write_full_joint_state,
)


@dataclass
class SMPDenoisingDiagnosticClip:
    dataset_path: Path
    source_name: str | None
    fps: float
    frame_range: tuple[int, int]
    noise_step: int
    raw_clip: PairedRawStateClip
    raw_joint_names: tuple[str, ...]
    decoded_joint_names: tuple[str, ...]
    clean_frames: torch.Tensor
    noisy_frames: torch.Tensor
    denoised_frames: torch.Tensor
    decoded_state_map: dict[str, PlaybackStateSequence]
    metrics: dict[str, float]


def select_playback_states(
    *,
    play_source: str,
    raw_clip: PairedRawStateClip,
    decoded_state_map: Mapping[str, PlaybackStateSequence],
) -> PairedRawStateClip | PlaybackStateSequence:
    if play_source == "raw":
        return raw_clip
    if play_source in decoded_state_map:
        return decoded_state_map[play_source]
    raise ValueError(
        "play_source must be one of {'raw', 'clean_decoded', 'noisy_decoded', 'denoised_decoded'}, "
        f"got {play_source!r}"
    )


def select_playback_joint_names(
    *,
    play_source: str,
    raw_joint_names: Sequence[str],
    decoded_joint_names: Sequence[str],
) -> tuple[str, ...]:
    if play_source == "raw":
        return tuple(str(name) for name in raw_joint_names)
    if play_source in {"clean_decoded", "noisy_decoded", "denoised_decoded"}:
        return tuple(str(name) for name in decoded_joint_names)
    raise ValueError(
        "play_source must be one of {'raw', 'clean_decoded', 'noisy_decoded', 'denoised_decoded'}, "
        f"got {play_source!r}"
    )


def should_use_direct_joint_write(
    *,
    joint_state_width: int,
    robot_joint_names: Sequence[str],
    playback_joint_names: Sequence[str],
    reference_joint_names: Sequence[str] | None = None,
    play_source: str | None = None,
) -> bool:
    use_full_joint_state = should_write_full_joint_state(
        joint_state_width=joint_state_width,
        robot_joint_count=len(tuple(robot_joint_names)),
    )
    if not use_full_joint_state:
        return False
    if play_source == "raw":
        return True
    if reference_joint_names is not None and tuple(str(name) for name in playback_joint_names) == tuple(
        str(name) for name in reference_joint_names
    ):
        return True
    return tuple(str(name) for name in playback_joint_names) == tuple(str(name) for name in robot_joint_names)


def prepare_smp_denoising_diagnostic_clip(
    *,
    dataset: str | Path,
    checkpoint: str | Path,
    source_name: str | None = None,
    style_id: int | None = None,
    clip_start_sec: float = 5.0,
    clip_duration_sec: float | None = 1.0,
    noise_step: int = 15,
    device: str | torch.device | None = None,
    stride: int = 1,
    root_body_index: int = 0,
    noise: torch.Tensor | None = None,
) -> SMPDenoisingDiagnosticClip:
    dataset_path = resolve_dataset_npz(dataset, source_name=source_name)
    payload = load_smp_paired_dataset(dataset_path)
    if "frames" not in payload or "fps" not in payload:
        raise KeyError("Paired dataset npz must include 'frames' and 'fps' fields")

    runtime_device = torch.device(device) if device is not None else torch.device("cpu")
    fps = _extract_scalar_field(payload["fps"], "fps")
    all_frames = torch.as_tensor(np.asarray(payload["frames"]), dtype=torch.float32, device=runtime_device)
    raw_clip = load_paired_raw_state_clip(
        dataset_path,
        clip_start_sec=clip_start_sec,
        clip_duration_sec=clip_duration_sec,
        device=runtime_device,
        root_body_index=root_body_index,
    )

    if clip_duration_sec is None:
        frame_range = raw_clip.frame_range
        clean_frames = all_frames[frame_range[0] : frame_range[1]]
    else:
        clean_frames, frame_range = slice_clip_frames(
            all_frames,
            fps=fps,
            clip_start_sec=clip_start_sec,
            clip_duration_sec=clip_duration_sec,
        )

    feature_block_offsets = read_feature_block_offsets(payload)
    raw_joint_names = tuple(read_replay_joint_names(payload))
    decoded_joint_names = _read_decoded_joint_names(payload)
    joint_axes = torch.as_tensor(np.asarray(payload["joint_axes"]), dtype=torch.float32, device=runtime_device)
    default_joint_pos = build_g1_default_joint_pos(decoded_joint_names).to(device=runtime_device)

    resolved_style_id = style_id if style_id is not None else _extract_optional_int_field(payload.get("style_id"), "style_id")
    loaded_prior = load_smp_prior_checkpoint(checkpoint, device=runtime_device, style_id=resolved_style_id)
    if int(clean_frames.shape[-1]) != int(loaded_prior.feature_dim):
        raise ValueError(
            f"Feature dim mismatch: dataset clip has {clean_frames.shape[-1]}, checkpoint expects {loaded_prior.feature_dim}"
        )

    windows, starts = build_motion_windows(clean_frames, window_size=loaded_prior.window_size, stride=stride)
    noise_step_value = int(noise_step)
    if noise_step_value < 0 or noise_step_value >= loaded_prior.num_diffusion_steps:
        raise ValueError(f"noise_step must be in [0, {loaded_prior.num_diffusion_steps - 1}], got {noise_step_value}")
    timesteps = torch.full((windows.shape[0],), noise_step_value, device=windows.device, dtype=torch.long)

    window_noise = torch.randn_like(windows) if noise is None else noise.to(device=windows.device, dtype=windows.dtype)
    if window_noise.shape != windows.shape:
        raise ValueError(f"noise must have shape {windows.shape}, got {window_noise.shape}")

    noisy_windows = loaded_prior.scheduler.q_sample(windows, timesteps, window_noise)

    style_tensor = None
    if loaded_prior.style_id is not None and getattr(loaded_prior.model, "num_styles", 0) > 0:
        style_tensor = torch.full((windows.shape[0],), int(loaded_prior.style_id), device=windows.device, dtype=torch.long)

    denoised_windows = denoise_motion_windows(
        windows,
        model=loaded_prior.model,
        scheduler=loaded_prior.scheduler,
        timesteps=timesteps,
        style_id=style_tensor,
        noise=window_noise,
    )

    total_frames = int(clean_frames.shape[0])
    noisy_frames = stitch_motion_windows(noisy_windows, starts, total_frames=total_frames)
    denoised_frames = stitch_motion_windows(denoised_windows, starts, total_frames=total_frames)
    dt = 1.0 / fps
    reference_root_pos_w = raw_clip.root_pos_w[0]
    decoded_state_map = {
        "clean_decoded": _apply_raw_root_carrier(
            reconstruct_playback_states(
                denoised_frame_features=clean_frames,
                feature_block_offsets=feature_block_offsets,
                joint_axes=joint_axes,
                default_joint_pos=default_joint_pos,
                reference_root_pos_w=reference_root_pos_w,
                dt=dt,
            ),
            raw_clip=raw_clip,
        ),
        "noisy_decoded": _apply_raw_root_carrier(
            reconstruct_playback_states(
                denoised_frame_features=noisy_frames,
                feature_block_offsets=feature_block_offsets,
                joint_axes=joint_axes,
                default_joint_pos=default_joint_pos,
                reference_root_pos_w=reference_root_pos_w,
                dt=dt,
            ),
            raw_clip=raw_clip,
        ),
        "denoised_decoded": _apply_raw_root_carrier(
            reconstruct_playback_states(
                denoised_frame_features=denoised_frames,
                feature_block_offsets=feature_block_offsets,
                joint_axes=joint_axes,
                default_joint_pos=default_joint_pos,
                reference_root_pos_w=reference_root_pos_w,
                dt=dt,
            ),
            raw_clip=raw_clip,
        ),
    }
    lin_start, lin_stop = feature_block_offsets["base_lin_vel_b"]
    ang_start, ang_stop = feature_block_offsets["base_ang_vel_b"]
    joint_start, joint_stop = feature_block_offsets["joint_rot6d_rel"]
    denoised_velocity = torch.cat((denoised_frames[:, lin_start:lin_stop], denoised_frames[:, ang_start:ang_stop]), dim=-1)
    clean_velocity = torch.cat((clean_frames[:, lin_start:lin_stop], clean_frames[:, ang_start:ang_stop]), dim=-1)
    metrics = {
        "clean_vs_noisy_mse": float((clean_frames - noisy_frames).pow(2).mean().item()),
        "clean_vs_denoised_mse": float((clean_frames - denoised_frames).pow(2).mean().item()),
        "joint_block_mse": float(
            (denoised_frames[:, joint_start:joint_stop] - clean_frames[:, joint_start:joint_stop]).pow(2).mean().item()
        ),
        "velocity_block_mse": float((denoised_velocity - clean_velocity).pow(2).mean().item()),
        "clean_decoded_vs_raw_joint_mae": float((decoded_state_map["clean_decoded"].joint_pos - raw_clip.joint_pos).abs().mean().item()),
        "clean_decoded_vs_raw_joint_max_abs": float(
            (decoded_state_map["clean_decoded"].joint_pos - raw_clip.joint_pos).abs().max().item()
        ),
        "clean_decoded_vs_raw_root_pos_mae": float(
            (decoded_state_map["clean_decoded"].root_pos_w - raw_clip.root_pos_w).abs().mean().item()
        ),
        "clean_decoded_vs_raw_root_quat_mae": float(
            (decoded_state_map["clean_decoded"].root_quat_w - raw_clip.root_quat_w).abs().mean().item()
        ),
    }
    resolved_source_name = source_name if source_name is not None else _extract_optional_str_field(payload.get("source_name"))

    return SMPDenoisingDiagnosticClip(
        dataset_path=dataset_path,
        source_name=resolved_source_name,
        fps=fps,
        frame_range=frame_range,
        noise_step=noise_step_value,
        raw_clip=raw_clip,
        raw_joint_names=raw_joint_names,
        decoded_joint_names=decoded_joint_names,
        clean_frames=clean_frames,
        noisy_frames=noisy_frames,
        denoised_frames=denoised_frames,
        decoded_state_map=decoded_state_map,
        metrics=metrics,
    )


def _extract_scalar_field(value: np.ndarray, field_name: str) -> float:
    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise ValueError(f"{field_name} must contain exactly one scalar value, got shape {np.asarray(value).shape}")
    return float(scalar[0])


def _extract_optional_int_field(value: object | None, field_name: str) -> int | None:
    if value is None:
        return None
    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise ValueError(f"{field_name} must contain exactly one scalar value, got shape {np.asarray(value).shape}")
    return int(scalar[0])


def _extract_optional_str_field(value: object | None) -> str | None:
    if value is None:
        return None
    items = np.asarray(value).reshape(-1)
    if items.size == 0:
        raise ValueError("source_name metadata must not be empty")
    resolved = str(items[0]).strip()
    if not resolved:
        raise ValueError("source_name metadata must not be blank")
    return resolved


def _read_decoded_joint_names(payload: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    if "smp_joint_names" in payload:
        return tuple(str(value) for value in np.asarray(payload["smp_joint_names"]).reshape(-1))
    if "joint_names" not in payload:
        raise KeyError("paired dataset must contain either 'smp_joint_names' or 'joint_names'")

    joint_names = tuple(str(value) for value in np.asarray(payload["joint_names"]).reshape(-1))
    joint_axes = np.asarray(payload["joint_axes"])
    if joint_axes.ndim != 2:
        raise ValueError(f"joint_axes must have shape (num_joints, 3), got {joint_axes.shape}")
    if len(joint_names) != int(joint_axes.shape[0]):
        raise KeyError("paired dataset is missing 'smp_joint_names' and raw joint_names do not match joint_axes")
    return joint_names


def _apply_raw_root_carrier(
    decoded_states: PlaybackStateSequence,
    *,
    raw_clip: PairedRawStateClip,
) -> PlaybackStateSequence:
    if decoded_states.root_pos_w.shape[0] != raw_clip.root_pos_w.shape[0]:
        raise ValueError(
            "decoded_states and raw_clip must have the same number of frames, "
            f"got {decoded_states.root_pos_w.shape[0]} and {raw_clip.root_pos_w.shape[0]}"
        )
    return PlaybackStateSequence(
        root_pos_w=raw_clip.root_pos_w.clone(),
        root_quat_w=raw_clip.root_quat_w.clone(),
        root_lin_vel_w=raw_clip.root_lin_vel_w.clone(),
        root_ang_vel_w=raw_clip.root_ang_vel_w.clone(),
        joint_pos=decoded_states.joint_pos,
        joint_vel=decoded_states.joint_vel,
    )
