"""Replay raw states stored in a paired SMP dataset npz."""

from __future__ import annotations

import argparse
import contextlib
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
    parser = argparse.ArgumentParser(description="Replay a paired SMP dataset using its saved raw states.")
    parser.add_argument("--dataset", default=None, help="Paired SMP dataset npz from csv_to_smp_paired_dataset.py.")
    parser.add_argument("--clip-start-sec", type=float, default=0.0, help="Playback start time in seconds.")
    parser.add_argument("--clip-duration-sec", type=float, default=None, help="Optional playback duration in seconds.")
    parser.add_argument("--root-body-index", type=int, default=0, help="Raw state body index used as root.")
    parser.add_argument("--play-once", action="store_true", default=False, help="Play once and exit.")
    parser.add_argument("--step-physics", action="store_true", default=False, help="Step physics instead of render-only replay.")
    parser.add_argument(
        "--wbt-root",
        default="/home/lucas/whole_body_tracking",
        help="whole_body_tracking repository root containing the G1 robot config.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_argparser()
args_cli = parser.parse_args()
if args_cli.dataset is None:
    parser.error("--dataset is required.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_rl.rsl_rl.smp_paired_dataset import (
    load_paired_raw_state_clip,
    load_smp_paired_dataset,
    read_feature_schema,
    read_replay_joint_names,
    should_write_full_joint_state,
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

    spec = importlib.util.spec_from_file_location("wbt_g1_cfg_smp_replay", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for WBT G1 config: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1_CYLINDER_CFG


def _design_scene() -> Articulation:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    robot_cfg = _load_wbt_robot_cfg(args_cli.wbt_root).replace(prim_path="/World/G1")
    return Articulation(robot_cfg)


def _load_joint_names(dataset_path: str | Path) -> list[str]:
    payload = load_smp_paired_dataset(dataset_path)
    return read_replay_joint_names(payload)


def _load_feature_summary(dataset_path: str | Path) -> tuple[str, int | None]:
    payload = load_smp_paired_dataset(dataset_path)
    feature_schema = read_feature_schema(payload)
    feature_dim = None
    if "frames" in payload:
        feature_dim = int(payload["frames"].shape[-1])
    return feature_schema, feature_dim


def _play_clip(sim: SimulationContext, robot: Articulation, clip, joint_names: list[str]) -> None:
    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos[0].clone()
    default_joint_vel = robot.data.default_joint_vel[0].clone()
    num_frames = int(clip.joint_pos.shape[0])
    use_full_joint_write = should_write_full_joint_state(
        joint_state_width=int(clip.joint_pos.shape[1]),
        robot_joint_count=int(default_joint_pos.shape[0]),
    )
    joint_ids_tensor = torch.tensor([], dtype=torch.long, device=sim.device)
    if not use_full_joint_write:
        joint_ids, resolved_names = robot.find_joints(joint_names, preserve_order=True)
        if resolved_names != joint_names:
            raise RuntimeError("Resolved robot joint order does not match dataset joint_names.")
        joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=sim.device)

    while simulation_app.is_running() and not simulation_app.is_exiting():
        for frame_idx in range(num_frames):
            if not simulation_app.is_running() or simulation_app.is_exiting():
                return

            if use_full_joint_write:
                full_joint_pos = clip.joint_pos[frame_idx]
                full_joint_vel = clip.joint_vel[frame_idx]
            else:
                full_joint_pos = default_joint_pos.clone()
                full_joint_vel = default_joint_vel.clone()
                full_joint_pos[joint_ids_tensor] = clip.joint_pos[frame_idx]
                full_joint_vel[joint_ids_tensor] = clip.joint_vel[frame_idx]

            root_state = torch.cat(
                (
                    clip.root_pos_w[frame_idx],
                    clip.root_quat_w[frame_idx],
                    clip.root_lin_vel_w[frame_idx],
                    clip.root_ang_vel_w[frame_idx],
                ),
                dim=-1,
            ).unsqueeze(0)
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(full_joint_pos.unsqueeze(0), full_joint_vel.unsqueeze(0))

            if args_cli.step_physics:
                robot.set_joint_position_target(full_joint_pos.unsqueeze(0))
                robot.write_data_to_sim()
                sim.step()
            else:
                sim.render()
            robot.update(sim_dt)

            pos_lookat = clip.root_pos_w[frame_idx].detach().cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.6]), pos_lookat)

        if args_cli.play_once:
            return


def main() -> None:
    clip = load_paired_raw_state_clip(
        args_cli.dataset,
        clip_start_sec=args_cli.clip_start_sec,
        clip_duration_sec=args_cli.clip_duration_sec,
        device=args_cli.device,
        root_body_index=args_cli.root_body_index,
    )
    joint_names = _load_joint_names(args_cli.dataset)
    feature_schema, feature_dim = _load_feature_summary(args_cli.dataset)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / clip.fps, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    robot = _design_scene()
    sim.reset()
    print("[SMP Replay] dataset:", args_cli.dataset)
    print("[SMP Replay] fps:", f"{clip.fps:.3f}")
    print("[SMP Replay] feature_schema:", feature_schema)
    if feature_dim is not None:
        print("[SMP Replay] feature_dim:", feature_dim)
    print("[SMP Replay] frame_range:", clip.frame_range)
    print("[SMP Replay] num_frames:", int(clip.joint_pos.shape[0]))
    print("[SMP Replay] mode:", "step_physics" if args_cli.step_physics else "render_only")
    print(
        "[SMP Replay] joint_write:",
        "full_raw_vector"
        if should_write_full_joint_state(
            joint_state_width=int(clip.joint_pos.shape[1]),
            robot_joint_count=int(robot.data.default_joint_pos.shape[1]),
        )
        else "mapped_subset",
    )
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        _play_clip(sim, robot, clip, joint_names)


if __name__ == "__main__":
    main()
    simulation_app.close()
