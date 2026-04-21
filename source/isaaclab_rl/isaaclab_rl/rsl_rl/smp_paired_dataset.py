from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


RAW_STATE_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

LEGACY_SMP_FEATURE_SCHEMA = "legacy_192"
EXTENDED_SMP_FEATURE_SCHEMA = "extended_198"

G1_MOTION_BODY_ORDER = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class FeatureErrorSummary:
    mean_abs_error: float
    max_abs_error: float


@dataclass(frozen=True)
class SMPFrameValidationSummary:
    mean_abs_error: float
    max_abs_error: float
    block_errors: dict[str, FeatureErrorSummary]


@dataclass(frozen=True)
class PairedRawStateClip:
    dataset_path: Path
    fps: float
    frame_range: tuple[int, int]
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


def _normalize_feature_schema(feature_schema: str) -> str:
    normalized = str(feature_schema)
    if normalized not in {LEGACY_SMP_FEATURE_SCHEMA, EXTENDED_SMP_FEATURE_SCHEMA}:
        raise ValueError(f"Unsupported SMP feature schema: {feature_schema}")
    return normalized


def g1_smp_feature_block_offsets(
    num_joints: int,
    num_end_effectors: int,
    *,
    feature_schema: str = LEGACY_SMP_FEATURE_SCHEMA,
) -> dict[str, tuple[int, int]]:
    if num_joints <= 0:
        raise ValueError(f"num_joints must be positive, got {num_joints}")
    if num_end_effectors < 0:
        raise ValueError(f"num_end_effectors must be non-negative, got {num_end_effectors}")

    feature_schema = _normalize_feature_schema(feature_schema)
    joint_stop = 6 + 6 * int(num_joints)
    feature_dim = joint_stop + 3 * int(num_end_effectors)
    offsets = {
        "base_lin_vel_b": (0, 3),
        "base_ang_vel_b": (3, 6),
        "joint_rot6d_rel": (6, joint_stop),
        "ee_pos_b": (joint_stop, feature_dim),
    }
    if feature_schema == EXTENDED_SMP_FEATURE_SCHEMA:
        offsets["base_lin_vel_w"] = (feature_dim, feature_dim + 3)
        offsets["base_ang_vel_w"] = (feature_dim + 3, feature_dim + 6)
    return offsets


def build_g1_default_joint_pos(joint_names: Sequence[str]) -> torch.Tensor:
    values = []
    for joint_name in joint_names:
        if joint_name in {"left_hip_pitch_joint", "right_hip_pitch_joint"}:
            values.append(-0.1)
        elif joint_name.endswith("_knee_joint"):
            values.append(0.3)
        elif joint_name.endswith("_ankle_pitch_joint"):
            values.append(-0.2)
        elif joint_name.endswith("_shoulder_pitch_joint"):
            values.append(0.3)
        elif joint_name == "left_shoulder_roll_joint":
            values.append(0.25)
        elif joint_name == "right_shoulder_roll_joint":
            values.append(-0.25)
        elif joint_name.endswith("_elbow_joint"):
            values.append(0.97)
        elif joint_name == "left_wrist_roll_joint":
            values.append(0.15)
        elif joint_name == "right_wrist_roll_joint":
            values.append(-0.15)
        else:
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def save_smp_paired_dataset(
    output_path: str | Path,
    *,
    frames: np.ndarray | torch.Tensor,
    fps: float,
    raw_states: Mapping[str, np.ndarray | torch.Tensor],
    joint_names: Sequence[str],
    joint_axes: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    ee_names: Sequence[str],
    feature_block_offsets: Mapping[str, tuple[int, int]],
    feature_schema: str = LEGACY_SMP_FEATURE_SCHEMA,
    window_size: int,
    stride: int,
    smp_joint_names: Sequence[str] | None = None,
    body_names: Sequence[str] | None = None,
    style_name: str | None = None,
    style_id: int | None = None,
    source_name: str | None = None,
    extra_fields: Mapping[str, np.ndarray | torch.Tensor] | None = None,
) -> Path:
    output_path = Path(output_path).expanduser()
    feature_schema = _normalize_feature_schema(feature_schema)
    frames_np = _as_float32_numpy(frames, "frames")
    if frames_np.ndim != 2:
        raise ValueError(f"Expected frames shape (num_frames, feature_dim), got {frames_np.shape}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    raw_payload = _validate_and_normalize_raw_states(raw_states, num_frames=int(frames_np.shape[0]))
    block_names, block_offsets = _normalize_feature_block_offsets(feature_block_offsets, feature_dim=int(frames_np.shape[1]))

    payload: dict[str, np.ndarray] = {
        "frames": frames_np,
        "fps": np.array([float(fps)], dtype=np.float32),
        "window_size": np.array([int(window_size)], dtype=np.int64),
        "stride": np.array([int(stride)], dtype=np.int64),
        "feature_dim": np.array([int(frames_np.shape[1])], dtype=np.int64),
        "feature_schema": np.asarray([feature_schema], dtype=np.str_),
        "joint_names": np.asarray(list(joint_names), dtype=np.str_),
        "joint_axes": _as_float32_numpy(joint_axes, "joint_axes"),
        "ee_names": np.asarray(list(ee_names), dtype=np.str_),
        "feature_block_names": np.asarray(block_names, dtype=np.str_),
        "feature_block_offsets": np.asarray(block_offsets, dtype=np.int64),
        **raw_payload,
    }
    if smp_joint_names is not None:
        payload["smp_joint_names"] = np.asarray(list(smp_joint_names), dtype=np.str_)
    if body_names is not None:
        payload["body_names"] = np.asarray(list(body_names), dtype=np.str_)
    if style_name is not None:
        payload["style_name"] = np.asarray([style_name], dtype=np.str_)
    if style_id is not None:
        payload["style_id"] = np.asarray([int(style_id)], dtype=np.int64)
    if source_name is not None:
        payload["source_name"] = np.asarray([source_name], dtype=np.str_)
    if extra_fields is not None:
        for key, value in extra_fields.items():
            if key in payload:
                raise KeyError(f"extra field {key!r} would overwrite a paired dataset field")
            payload[key] = _to_numpy(value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path


def load_smp_paired_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"SMP paired dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def read_feature_schema(payload: Mapping[str, np.ndarray]) -> str:
    if "feature_schema" in payload:
        value = np.asarray(payload["feature_schema"]).reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if hasattr(value, "item"):
            value = value.item()
        return _normalize_feature_schema(str(value))
    if "frames" not in payload:
        raise KeyError("paired dataset must contain feature_schema or frames")
    frames = np.asarray(payload["frames"])
    if frames.ndim != 2:
        raise ValueError(f"Expected frames shape (num_frames, feature_dim), got {frames.shape}")
    if frames.shape[-1] == 198:
        return EXTENDED_SMP_FEATURE_SCHEMA
    if frames.shape[-1] == 192:
        return LEGACY_SMP_FEATURE_SCHEMA
    raise ValueError(
        "Cannot infer SMP feature_schema from feature_dim="
        f"{frames.shape[-1]}; expected explicit feature_schema metadata"
    )


def read_feature_block_offsets(payload: Mapping[str, np.ndarray]) -> dict[str, tuple[int, int]]:
    if "feature_block_names" not in payload or "feature_block_offsets" not in payload:
        raise KeyError("paired dataset must contain feature_block_names and feature_block_offsets")
    names = [str(value) for value in np.asarray(payload["feature_block_names"]).reshape(-1)]
    offsets = np.asarray(payload["feature_block_offsets"], dtype=np.int64)
    if offsets.shape != (len(names), 2):
        raise ValueError(f"feature_block_offsets must have shape ({len(names)}, 2), got {offsets.shape}")
    return {name: (int(offset[0]), int(offset[1])) for name, offset in zip(names, offsets)}


def read_replay_joint_names(payload: Mapping[str, np.ndarray]) -> list[str]:
    if "joint_names" not in payload:
        raise KeyError("paired dataset must contain joint_names for raw replay order")
    return [str(value) for value in np.asarray(payload["joint_names"]).reshape(-1)]


def should_write_full_joint_state(*, joint_state_width: int, robot_joint_count: int) -> bool:
    if joint_state_width <= 0:
        raise ValueError(f"joint_state_width must be positive, got {joint_state_width}")
    if robot_joint_count <= 0:
        raise ValueError(f"robot_joint_count must be positive, got {robot_joint_count}")
    return int(joint_state_width) == int(robot_joint_count)


def validate_smp_frames_match(
    actual_frames: np.ndarray | torch.Tensor,
    expected_frames: np.ndarray | torch.Tensor,
    *,
    feature_block_offsets: Mapping[str, tuple[int, int]] | None = None,
) -> SMPFrameValidationSummary:
    actual = torch.as_tensor(actual_frames, dtype=torch.float32)
    expected = torch.as_tensor(expected_frames, dtype=torch.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"actual and expected frames must share shape, got {tuple(actual.shape)} and {tuple(expected.shape)}")
    if actual.ndim != 2:
        raise ValueError(f"Expected frames shape (num_frames, feature_dim), got {tuple(actual.shape)}")

    abs_error = (actual - expected).abs()
    block_errors: dict[str, FeatureErrorSummary] = {}
    if feature_block_offsets is not None:
        for block_name, (start, stop) in feature_block_offsets.items():
            if start < 0 or stop <= start or stop > actual.shape[-1]:
                raise ValueError(f"Invalid feature block {block_name!r}: ({start}, {stop})")
            block_error = abs_error[:, start:stop]
            block_errors[str(block_name)] = FeatureErrorSummary(
                mean_abs_error=float(block_error.mean().item()),
                max_abs_error=float(block_error.max().item()),
            )

    return SMPFrameValidationSummary(
        mean_abs_error=float(abs_error.mean().item()),
        max_abs_error=float(abs_error.max().item()),
        block_errors=block_errors,
    )


def format_validation_summary(summary: SMPFrameValidationSummary) -> str:
    lines = [
        f"mean_abs_error: {summary.mean_abs_error:.8f}",
        f"max_abs_error: {summary.max_abs_error:.8f}",
    ]
    for block_name, block_summary in summary.block_errors.items():
        lines.append(
            f"{block_name}: mean_abs_error={block_summary.mean_abs_error:.8f}, "
            f"max_abs_error={block_summary.max_abs_error:.8f}"
        )
    return "\n".join(lines)


def load_paired_raw_state_clip(
    dataset_path: str | Path,
    *,
    clip_start_sec: float = 0.0,
    clip_duration_sec: float | None = None,
    device: str | torch.device | None = None,
    root_body_index: int = 0,
) -> PairedRawStateClip:
    payload = load_smp_paired_dataset(dataset_path)
    missing = [key for key in ("fps", *RAW_STATE_KEYS) if key not in payload]
    if missing:
        raise KeyError(f"SMP paired dataset is missing raw replay fields: {missing}")

    fps = _extract_scalar(payload["fps"], "fps")
    joint_pos = torch.as_tensor(payload["joint_pos"], dtype=torch.float32)
    joint_vel = torch.as_tensor(payload["joint_vel"], dtype=torch.float32)
    body_pos_w = torch.as_tensor(payload["body_pos_w"], dtype=torch.float32)
    body_quat_w = torch.as_tensor(payload["body_quat_w"], dtype=torch.float32)
    body_lin_vel_w = torch.as_tensor(payload["body_lin_vel_w"], dtype=torch.float32)
    body_ang_vel_w = torch.as_tensor(payload["body_ang_vel_w"], dtype=torch.float32)
    _validate_raw_state_tensors(joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w)

    if root_body_index < 0 or root_body_index >= int(body_pos_w.shape[1]):
        raise ValueError(f"root_body_index={root_body_index} is out of range for num_bodies={int(body_pos_w.shape[1])}")

    start_idx, end_idx = slice_frame_range(
        total_frames=int(joint_pos.shape[0]),
        fps=fps,
        clip_start_sec=clip_start_sec,
        clip_duration_sec=clip_duration_sec,
    )
    runtime_device = torch.device(device) if device is not None else torch.device("cpu")
    return PairedRawStateClip(
        dataset_path=Path(dataset_path).expanduser(),
        fps=fps,
        frame_range=(start_idx, end_idx),
        root_pos_w=body_pos_w[start_idx:end_idx, root_body_index].to(runtime_device),
        root_quat_w=body_quat_w[start_idx:end_idx, root_body_index].to(runtime_device),
        root_lin_vel_w=body_lin_vel_w[start_idx:end_idx, root_body_index].to(runtime_device),
        root_ang_vel_w=body_ang_vel_w[start_idx:end_idx, root_body_index].to(runtime_device),
        joint_pos=joint_pos[start_idx:end_idx].to(runtime_device),
        joint_vel=joint_vel[start_idx:end_idx].to(runtime_device),
    )


def slice_frame_range(
    *,
    total_frames: int,
    fps: float,
    clip_start_sec: float = 0.0,
    clip_duration_sec: float | None = None,
) -> tuple[int, int]:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if clip_start_sec < 0.0:
        raise ValueError(f"clip_start_sec must be non-negative, got {clip_start_sec}")

    start_idx = int(clip_start_sec * fps)
    if clip_duration_sec is None:
        end_idx = total_frames
    else:
        if clip_duration_sec <= 0.0:
            raise ValueError(f"clip_duration_sec must be positive, got {clip_duration_sec}")
        duration_frames = int(clip_duration_sec * fps)
        if duration_frames <= 0:
            raise ValueError("Requested clip duration resolves to zero frames")
        end_idx = start_idx + duration_frames

    if start_idx < 0 or start_idx >= total_frames or end_idx > total_frames:
        raise ValueError(f"Requested clip [{start_idx}, {end_idx}) is out of range for total_frames={total_frames}")
    return start_idx, end_idx


def build_g1_smp_frames_from_raw_motion(
    raw_motion: Mapping[str, np.ndarray | torch.Tensor],
    *,
    body_order: Sequence[str] = G1_MOTION_BODY_ORDER,
    joint_names: Sequence[str],
    joint_axes: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    default_joint_pos: np.ndarray | torch.Tensor | None = None,
    ee_names: Sequence[str],
    expected_feature_dim: int | None = None,
    feature_schema: str = LEGACY_SMP_FEATURE_SCHEMA,
) -> torch.Tensor:
    missing = [key for key in ("joint_pos", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w") if key not in raw_motion]
    if missing:
        raise KeyError(f"raw motion is missing required fields: {missing}")

    joint_pos = torch.as_tensor(raw_motion["joint_pos"], dtype=torch.float32)
    body_pos_w = torch.as_tensor(raw_motion["body_pos_w"], dtype=torch.float32)
    body_quat_w = torch.as_tensor(raw_motion["body_quat_w"], dtype=torch.float32)
    body_lin_vel_w = torch.as_tensor(raw_motion["body_lin_vel_w"], dtype=torch.float32)
    body_ang_vel_w = torch.as_tensor(raw_motion["body_ang_vel_w"], dtype=torch.float32)

    if joint_pos.ndim != 2:
        raise ValueError(f"Expected joint_pos shape (num_frames, num_joints), got {tuple(joint_pos.shape)}")
    if joint_pos.shape[-1] != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joints, got {joint_pos.shape[-1]}")
    _validate_body_tensor("body_pos_w", body_pos_w, int(joint_pos.shape[0]), 3)
    _validate_body_tensor("body_quat_w", body_quat_w, int(joint_pos.shape[0]), 4)
    _validate_body_tensor("body_lin_vel_w", body_lin_vel_w, int(joint_pos.shape[0]), 3)
    _validate_body_tensor("body_ang_vel_w", body_ang_vel_w, int(joint_pos.shape[0]), 3)
    if body_pos_w.shape[1] != len(body_order):
        raise ValueError(f"Expected body_order length {body_pos_w.shape[1]}, got {len(body_order)}")

    body_index = {body_name: index for index, body_name in enumerate(body_order)}
    if "pelvis" not in body_index:
        raise KeyError("body_order must include 'pelvis'")
    missing_ee = [name for name in ee_names if name not in body_index]
    if missing_ee:
        raise KeyError(f"body_order is missing end-effector bodies: {missing_ee}")

    root_body_index = body_index["pelvis"]
    ee_body_ids = [body_index[name] for name in ee_names]
    joint_axes_tensor = torch.as_tensor(joint_axes, dtype=torch.float32, device=joint_pos.device)
    if default_joint_pos is None:
        default_joint_pos_tensor = build_g1_default_joint_pos(joint_names).to(device=joint_pos.device)
    else:
        default_joint_pos_tensor = torch.as_tensor(default_joint_pos, dtype=torch.float32, device=joint_pos.device)

    components = build_smp_feature_components(
        root_pos_w=body_pos_w[:, root_body_index],
        root_quat_w=body_quat_w[:, root_body_index],
        root_lin_vel_w=body_lin_vel_w[:, root_body_index],
        root_ang_vel_w=body_ang_vel_w[:, root_body_index],
        joint_pos=joint_pos,
        default_joint_pos=default_joint_pos_tensor,
        joint_axes=joint_axes_tensor,
        ee_pos_w=body_pos_w[:, ee_body_ids],
    )
    return pack_smp_frame_features(
        base_lin_vel_b=components["base_lin_vel_b"],
        base_ang_vel_b=components["base_ang_vel_b"],
        joint_rot6d_rel=components["joint_rot6d_rel"],
        ee_pos_b=components["ee_pos_b"],
        base_lin_vel_w=components["base_lin_vel_w"],
        base_ang_vel_w=components["base_ang_vel_w"],
        feature_schema=feature_schema,
        expected_feature_dim=expected_feature_dim,
    )


def build_smp_feature_components(
    *,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_axes: torch.Tensor,
    ee_pos_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if root_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected root_pos_w last dim 3, got {root_pos_w.shape[-1]}")
    if root_quat_w.shape[-1] != 4:
        raise ValueError(f"Expected root_quat_w last dim 4, got {root_quat_w.shape[-1]}")
    if root_lin_vel_w.shape[-1] != 3 or root_ang_vel_w.shape[-1] != 3:
        raise ValueError("Expected root linear/angular velocities to have last dim 3")
    if ee_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected ee_pos_w last dim 3, got {ee_pos_w.shape[-1]}")

    heading_rotation = build_heading_frame_rotation(root_quat_w)
    ee_pos_rel_w = ee_pos_w - root_pos_w.unsqueeze(-2)
    return {
        "base_lin_vel_b": world_to_local_frame(heading_rotation, root_lin_vel_w),
        "base_ang_vel_b": world_to_local_frame(heading_rotation, root_ang_vel_w),
        "joint_rot6d_rel": joint_positions_to_rot6d(joint_pos, default_joint_pos, joint_axes),
        "ee_pos_b": world_to_local_frame(heading_rotation, ee_pos_rel_w),
        "base_lin_vel_w": root_lin_vel_w,
        "base_ang_vel_w": root_ang_vel_w,
    }


def pack_smp_frame_features(
    *,
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    joint_rot6d_rel: torch.Tensor,
    ee_pos_b: torch.Tensor,
    base_lin_vel_w: torch.Tensor | None = None,
    base_ang_vel_w: torch.Tensor | None = None,
    feature_schema: str = LEGACY_SMP_FEATURE_SCHEMA,
    expected_feature_dim: int | None = None,
) -> torch.Tensor:
    feature_schema = _normalize_feature_schema(feature_schema)
    lead_shape = base_lin_vel_b.shape[:-1]
    if base_lin_vel_b.shape[-1] != 3:
        raise ValueError(f"Expected base_lin_vel_b last dim 3, got {base_lin_vel_b.shape[-1]}")
    if base_ang_vel_b.shape[-1] != 3:
        raise ValueError(f"Expected base_ang_vel_b last dim 3, got {base_ang_vel_b.shape[-1]}")
    if base_ang_vel_b.shape[:-1] != lead_shape:
        raise ValueError("SMP root velocity feature inputs have inconsistent leading dimensions")
    if joint_rot6d_rel.shape[:-2] != lead_shape or ee_pos_b.shape[:-2] != lead_shape:
        raise ValueError("SMP joint rot6d or end-effector inputs have inconsistent leading dimensions")
    if joint_rot6d_rel.shape[-1] != 6:
        raise ValueError(f"Expected joint_rot6d_rel last dim 6, got {joint_rot6d_rel.shape[-1]}")
    if ee_pos_b.shape[-1] != 3:
        raise ValueError(f"Expected ee_pos_b last dim 3, got {ee_pos_b.shape[-1]}")

    feature_parts = [
        base_lin_vel_b,
        base_ang_vel_b,
        joint_rot6d_rel.reshape(*lead_shape, -1),
        ee_pos_b.reshape(*lead_shape, -1),
    ]
    if feature_schema == EXTENDED_SMP_FEATURE_SCHEMA:
        if base_lin_vel_w is None or base_ang_vel_w is None:
            raise ValueError("extended_198 schema requires base_lin_vel_w and base_ang_vel_w")
        if base_lin_vel_w.shape[:-1] != lead_shape or base_ang_vel_w.shape[:-1] != lead_shape:
            raise ValueError("SMP world velocity feature inputs have inconsistent leading dimensions")
        if base_lin_vel_w.shape[-1] != 3 or base_ang_vel_w.shape[-1] != 3:
            raise ValueError("Expected SMP world linear/angular velocities to have last dim 3")
        feature_parts.extend([base_lin_vel_w, base_ang_vel_w])
    features = torch.cat(feature_parts, dim=-1)
    if expected_feature_dim is not None and features.shape[-1] != expected_feature_dim:
        raise ValueError(f"Expected SMP feature dim {expected_feature_dim}, got {features.shape[-1]}")
    return features


def joint_positions_to_rot6d(joint_pos: torch.Tensor, default_joint_pos: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    if joint_pos.shape[-1] != joint_axes.shape[0]:
        raise ValueError(f"Expected joint_pos last dim {joint_axes.shape[0]}, got {joint_pos.shape[-1]}")
    return joint_angle_offsets_to_rot6d(joint_pos - default_joint_pos, joint_axes)


def joint_angle_offsets_to_rot6d(joint_angle_offsets: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    if joint_axes.ndim != 2 or joint_axes.shape[-1] != 3:
        raise ValueError(f"Expected joint_axes shape (num_joints, 3), got {tuple(joint_axes.shape)}")
    if joint_angle_offsets.shape[-1] != joint_axes.shape[0]:
        raise ValueError(f"Expected joint_angle_offsets last dim {joint_axes.shape[0]}, got {joint_angle_offsets.shape[-1]}")

    normalized_axes = _normalize(joint_axes.to(device=joint_angle_offsets.device, dtype=joint_angle_offsets.dtype))
    axis_shape = (1,) * (joint_angle_offsets.ndim - 1) + normalized_axes.shape
    expanded_axes = normalized_axes.view(axis_shape)
    half_angle = 0.5 * joint_angle_offsets
    quat_wxyz = torch.cat(
        (
            torch.cos(half_angle).unsqueeze(-1),
            expanded_axes * torch.sin(half_angle).unsqueeze(-1),
        ),
        dim=-1,
    )
    return quat_to_rot6d(quat_wxyz)


def quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quat_wxyz last dim 4, got {quat_wxyz.shape[-1]}")
    return _matrix_from_quat(quat_wxyz)[..., :2].reshape(*quat_wxyz.shape[:-1], 6)


def build_heading_frame_rotation(root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    if root_quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected root_quat_wxyz last dim 4, got {root_quat_wxyz.shape[-1]}")

    root_rotation = _matrix_from_quat(root_quat_wxyz)
    forward_w = root_rotation[..., :, 0]
    fallback_w = root_rotation[..., :, 1]
    up_w = torch.zeros_like(forward_w)
    up_w[..., 2] = 1.0

    forward_proj = forward_w - (forward_w * up_w).sum(dim=-1, keepdim=True) * up_w
    fallback_proj = fallback_w - (fallback_w * up_w).sum(dim=-1, keepdim=True) * up_w
    use_fallback = forward_proj.norm(dim=-1, keepdim=True) < 1.0e-6
    x_axis_w = _normalize(torch.where(use_fallback, fallback_proj, forward_proj))
    y_axis_w = up_w
    z_axis_w = _normalize(torch.cross(x_axis_w, y_axis_w, dim=-1))
    x_axis_w = _normalize(torch.cross(y_axis_w, z_axis_w, dim=-1))
    return torch.stack((x_axis_w, y_axis_w, z_axis_w), dim=-1)


def world_to_local_frame(rotation_world_from_local: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    if rotation_world_from_local.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation_world_from_local trailing shape (3, 3), got {rotation_world_from_local.shape[-2:]}"
        )
    if vec_w.shape[-1] != 3:
        raise ValueError(f"Expected vec_w last dim 3, got {vec_w.shape[-1]}")
    while rotation_world_from_local.ndim < vec_w.ndim + 1:
        rotation_world_from_local = rotation_world_from_local.unsqueeze(-3)
    return torch.matmul(vec_w.unsqueeze(-2), rotation_world_from_local).squeeze(-2)


def _matrix_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = _normalize(quat_wxyz)
    w, x, y, z = quat_wxyz.unbind(dim=-1)
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


def _normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def _validate_and_normalize_raw_states(
    raw_states: Mapping[str, np.ndarray | torch.Tensor],
    *,
    num_frames: int,
) -> dict[str, np.ndarray]:
    missing = [key for key in RAW_STATE_KEYS if key not in raw_states]
    if missing:
        raise KeyError(f"raw_states is missing required fields: {missing}")

    payload = {key: _as_float32_numpy(raw_states[key], key) for key in RAW_STATE_KEYS}
    if payload["joint_pos"].ndim != 2:
        raise ValueError(f"Expected joint_pos shape (num_frames, num_joints), got {payload['joint_pos'].shape}")
    if payload["joint_vel"].shape != payload["joint_pos"].shape:
        raise ValueError("joint_pos and joint_vel must have matching shape")
    if payload["joint_pos"].shape[0] != num_frames:
        raise ValueError(f"raw state frame count {payload['joint_pos'].shape[0]} does not match frames count {num_frames}")
    for key, width in (
        ("body_pos_w", 3),
        ("body_quat_w", 4),
        ("body_lin_vel_w", 3),
        ("body_ang_vel_w", 3),
    ):
        value = payload[key]
        if value.ndim != 3 or value.shape[0] != num_frames or value.shape[-1] != width:
            raise ValueError(f"Expected {key} shape (num_frames, num_bodies, {width}), got {value.shape}")
    return payload


def _validate_raw_state_tensors(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
) -> None:
    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise ValueError("joint_pos and joint_vel must have matching shape (num_frames, num_joints)")
    for field_name, tensor, width in (
        ("body_pos_w", body_pos_w, 3),
        ("body_quat_w", body_quat_w, 4),
        ("body_lin_vel_w", body_lin_vel_w, 3),
        ("body_ang_vel_w", body_ang_vel_w, 3),
    ):
        if tensor.ndim != 3 or tensor.shape[0] != joint_pos.shape[0] or tensor.shape[-1] != width:
            raise ValueError(f"{field_name} must have shape (num_frames, num_bodies, {width}), got {tuple(tensor.shape)}")


def _validate_body_tensor(name: str, tensor: torch.Tensor, num_frames: int, width: int) -> None:
    if tensor.ndim != 3 or tensor.shape[0] != num_frames or tensor.shape[-1] != width:
        raise ValueError(f"Expected {name} shape ({num_frames}, num_bodies, {width}), got {tuple(tensor.shape)}")


def _normalize_feature_block_offsets(
    feature_block_offsets: Mapping[str, tuple[int, int]],
    *,
    feature_dim: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    names: list[str] = []
    offsets: list[tuple[int, int]] = []
    for block_name, block in feature_block_offsets.items():
        if len(block) != 2:
            raise ValueError(f"feature block {block_name!r} must be a (start, stop) pair")
        start, stop = int(block[0]), int(block[1])
        if start < 0 or stop <= start or stop > feature_dim:
            raise ValueError(f"Invalid feature block {block_name!r}: ({start}, {stop})")
        names.append(str(block_name))
        offsets.append((start, stop))
    return names, offsets


def _extract_scalar(value: np.ndarray, field_name: str) -> float:
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Dataset metadata field {field_name!r} is empty")
    scalar = float(values[0])
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"Dataset metadata field {field_name!r} must be positive and finite, got {scalar}")
    return scalar


def _as_float32_numpy(value: np.ndarray | torch.Tensor | Sequence[float], field_name: str) -> np.ndarray:
    array = _to_numpy(value).astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains non-finite values")
    return array


def _to_numpy(value: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
