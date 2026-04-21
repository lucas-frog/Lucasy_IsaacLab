"""Convert a LAFAN-style CSV motion to a paired SMP training/replay dataset."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types

from path_bootstrap import prepend_local_source_paths

prepend_local_source_paths(__file__)

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:
    if exc.name == "isaacsim" and ("--help" in sys.argv or "-h" in sys.argv):
        isaacsim_stub = types.ModuleType("isaacsim")

        class _SimulationApp:
            pass

        isaacsim_stub.SimulationApp = _SimulationApp
        sys.modules["isaacsim"] = isaacsim_stub
        from isaaclab.app import AppLauncher
    else:
        raise


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a LAFAN CSV file to a paired SMP dataset npz.")
    parser.add_argument("--input-file", default=None, help="Input LAFAN CSV path.")
    parser.add_argument("--input-fps", type=int, default=30, help="Input CSV FPS.")
    parser.add_argument("--output", default=None, help="Output paired dataset npz path.")
    parser.add_argument("--output-fps", type=int, default=50, help="Output dataset FPS.")
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional inclusive 1-based input frame range.",
    )
    parser.add_argument("--window-size", type=int, default=None, help="SMP metadata window size.")
    parser.add_argument("--stride", type=int, default=1, help="SMP metadata stride.")
    parser.add_argument("--style-name", default=None, help="Optional style name metadata.")
    parser.add_argument("--style-id", type=int, default=None, help="Optional style id metadata.")
    parser.add_argument("--source-name", default=None, help="Optional source clip name metadata.")
    parser.add_argument(
        "--feature-schema",
        choices=("legacy_192", "extended_198"),
        default="legacy_192",
        help="SMP feature layout to export. extended_198 appends base_lin_vel_w and base_ang_vel_w.",
    )
    parser.add_argument(
        "--wbt-root",
        default="/home/lucas/whole_body_tracking",
        help="whole_body_tracking repository root containing the G1 robot config.",
    )
    parser.add_argument(
        "--compare-dataset",
        default=None,
        help="Optional existing SMP dataset npz whose frames should match this export.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_argparser()
args_cli = parser.parse_args()
if args_cli.input_file is None:
    parser.error("--input-file is required.")
if args_cli.output is None:
    parser.error("--output is required.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
from isaaclab_rl.rsl_rl.smp_paired_dataset import (
    build_g1_smp_frames_from_raw_motion,
    format_validation_summary,
    g1_smp_feature_block_offsets,
    load_smp_paired_dataset,
    save_smp_paired_dataset,
    validate_smp_frames_match,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.config import (
    g1_ee_names,
    g1_smp_feature_dim_for_schema,
    g1_smp_joint_axes,
    g1_smp_joint_names,
    g1_smp_window_size,
)


def _load_wbt_robot_cfg(wbt_root: str | Path):
    module_path = Path(wbt_root).expanduser().resolve() / "source" / "whole_body_tracking" / "whole_body_tracking" / "robots" / "g1.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Could not find WBT G1 config: {module_path}")

    package_root = module_path.parent.parent
    if "whole_body_tracking" not in sys.modules:
        package = types.ModuleType("whole_body_tracking")
        package.__path__ = [str(package_root)]
        sys.modules["whole_body_tracking"] = package
    if "whole_body_tracking.assets" not in sys.modules:
        assets_module = types.ModuleType("whole_body_tracking.assets")
        assets_module.ASSET_DIR = str(package_root / "assets")
        sys.modules["whole_body_tracking.assets"] = assets_module
    if "whole_body_tracking.robots" not in sys.modules:
        robots_package = types.ModuleType("whole_body_tracking.robots")
        robots_package.__path__ = [str(module_path.parent)]
        sys.modules["whole_body_tracking.robots"] = robots_package

    spec = importlib.util.spec_from_file_location("wbt_g1_cfg_csv_to_smp", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for WBT G1 config: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1_CYLINDER_CFG


@configclass
class ExportSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = _load_wbt_robot_cfg(args_cli.wbt_root).replace(prim_path="{ENV_REGEX_NS}/Robot")


class CSVInterpolatedMotion:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = int(input_fps)
        self.output_fps = int(output_fps)
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.device = device
        self.frame_range = frame_range
        self.current_idx = 0
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            start, end = self.frame_range
            motion = torch.from_numpy(
                np.loadtxt(self.motion_file, delimiter=",", skiprows=start - 1, max_rows=end - start + 1)
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_pos_input = motion[:, :3]
        self.motion_base_rot_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_pos_input = motion[:, 7:]
        self.input_frames = int(motion.shape[0])
        self.duration = (self.input_frames - 1) * self.input_dt
        if self.input_frames < 3:
            raise ValueError(f"Expected at least 3 input frames, got {self.input_frames}")
        print(f"[CSV] loaded {self.motion_file}: frames={self.input_frames}, duration={self.duration:.3f}s")

    def _interpolate_motion(self) -> None:
        times = torch.arange(0.0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = int(times.shape[0])
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_pos = self._lerp(
            self.motion_base_pos_input[index_0],
            self.motion_base_pos_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rot = self._slerp(
            self.motion_base_rot_input[index_0],
            self.motion_base_rot_input[index_1],
            blend,
        )
        self.motion_dof_pos = self._lerp(
            self.motion_dof_pos_input[index_0],
            self.motion_dof_pos_input[index_1],
            blend.unsqueeze(1),
        )
        print(f"[CSV] interpolated to frames={self.output_frames}, fps={self.output_fps}")

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1.0 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(a)
        for index in range(a.shape[0]):
            output[index] = quat_slerp(a[index], b[index], blend[index])
        return output

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vel = torch.gradient(self.motion_base_pos, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vel = torch.gradient(self.motion_dof_pos, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vel = self._so3_derivative(self.motion_base_rot, self.output_dt)

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_pos[self.current_idx : self.current_idx + 1],
            self.motion_base_rot[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vel[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vel[self.current_idx : self.current_idx + 1],
            self.motion_dof_pos[self.current_idx : self.current_idx + 1],
            self.motion_dof_vel[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset = True
        return state, reset


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.unsqueeze(1), v.unsqueeze(-1)).squeeze(-1) * 2.0
    return a - b + c


def _capture_raw_states(sim: SimulationContext, scene: InteractiveScene) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    motion = CSVInterpolatedMotion(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
    )
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(g1_smp_joint_names, preserve_order=True)[0]
    raw_joint_names = list(robot.data.joint_names)
    raw_body_names = list(robot.data.body_names)
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=sim.device).expand(scene.num_envs, 3)
    log: dict[str, list[np.ndarray]] = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "projected_gravity": [],
    }

    while simulation_app.is_running() and not simulation_app.is_exiting():
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ), reset = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        projected_gravity = quat_rotate_inverse(robot.data.root_quat_w, gravity_vec_w)
        log["projected_gravity"].append(projected_gravity[0].cpu().numpy().copy())
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        if reset:
            break

    return {key: np.stack(value, axis=0) for key, value in log.items()}, raw_joint_names, raw_body_names


def _save_dataset(raw_states: dict[str, np.ndarray], raw_joint_names: list[str], raw_body_names: list[str]) -> Path:
    feature_schema = str(args_cli.feature_schema)
    expected_feature_dim = g1_smp_feature_dim_for_schema(feature_schema)
    feature_block_offsets = g1_smp_feature_block_offsets(
        len(g1_smp_joint_names),
        len(g1_ee_names),
        feature_schema=feature_schema,
    )
    frames = build_g1_smp_frames_from_raw_motion(
        raw_states,
        body_order=raw_body_names,
        joint_names=g1_smp_joint_names,
        joint_axes=g1_smp_joint_axes,
        ee_names=g1_ee_names,
        expected_feature_dim=expected_feature_dim,
        feature_schema=feature_schema,
    )
    output_path = save_smp_paired_dataset(
        args_cli.output,
        frames=frames,
        fps=args_cli.output_fps,
        raw_states=raw_states,
        joint_names=raw_joint_names,
        joint_axes=g1_smp_joint_axes,
        ee_names=g1_ee_names,
        feature_block_offsets=feature_block_offsets,
        feature_schema=feature_schema,
        window_size=args_cli.window_size if args_cli.window_size is not None else g1_smp_window_size,
        stride=args_cli.stride,
        smp_joint_names=g1_smp_joint_names,
        body_names=raw_body_names,
        style_name=args_cli.style_name,
        style_id=args_cli.style_id,
        source_name=args_cli.source_name,
        extra_fields={"projected_gravity": raw_states["projected_gravity"]},
    )

    recomputed = build_g1_smp_frames_from_raw_motion(
        raw_states,
        body_order=raw_body_names,
        joint_names=g1_smp_joint_names,
        joint_axes=g1_smp_joint_axes,
        ee_names=g1_ee_names,
        expected_feature_dim=expected_feature_dim,
        feature_schema=feature_schema,
    )
    validation = validate_smp_frames_match(frames, recomputed, feature_block_offsets=feature_block_offsets)
    print("[SMP] self validation")
    print(format_validation_summary(validation))

    if args_cli.compare_dataset is not None:
        reference_payload = load_smp_paired_dataset(args_cli.compare_dataset)
        if "frames" not in reference_payload:
            raise KeyError(f"compare dataset is missing 'frames': {args_cli.compare_dataset}")
        comparison = validate_smp_frames_match(
            frames,
            reference_payload["frames"],
            feature_block_offsets=feature_block_offsets,
        )
        print("[SMP] comparison against existing dataset")
        print(format_validation_summary(comparison))

    print(f"[SMP] saved paired dataset: {output_path}")
    print(f"[SMP] feature_schema: {feature_schema}")
    print(f"[SMP] frames: {tuple(frames.shape)}")
    if raw_joint_names != list(g1_smp_joint_names):
        print("[SMP] raw replay joint order differs from SMP feature joint order; saved both metadata fields.")
    return output_path


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ExportSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("[SMP] capture started")
    with torch.inference_mode():
        raw_states, raw_joint_names, raw_body_names = _capture_raw_states(sim, scene)
    _save_dataset(raw_states, raw_joint_names, raw_body_names)


if __name__ == "__main__":
    main()
    simulation_app.close()
