# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay SMP denoising diagnostics on a single G1 robot in Isaac Sim."""

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
    parser = argparse.ArgumentParser(description="Replay denoising diagnostics from a paired SMP dataset npz.")
    parser.add_argument("--dataset", default=None, help="Paired SMP dataset npz path.")
    parser.add_argument("--checkpoint", default=None, help="SMP prior checkpoint path.")
    parser.add_argument("--clip-start-sec", type=float, default=5.0, help="Playback start time in seconds.")
    parser.add_argument("--clip-duration-sec", type=float, default=1.0, help="Playback duration in seconds.")
    parser.add_argument("--noise-step", type=int, default=15, help="Diffusion timestep used for forward noising.")
    parser.add_argument(
        "--play-source",
        default="denoised_decoded",
        choices=("raw", "clean_decoded", "noisy_decoded", "denoised_decoded"),
        help="Which state source to replay in Isaac Sim.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride for denoising.")
    parser.add_argument("--style-id", type=int, default=None, help="Optional style id override for the checkpoint.")
    parser.add_argument("--save-debug-npz", default=None, help="Optional path to save clean/noisy/denoised frames.")
    parser.add_argument("--play-once", action="store_true", default=False, help="Play one pass then exit.")
    parser.add_argument("--step-physics", action="store_true", default=False, help="Step physics instead of render-only playback.")
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
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_rl.rsl_rl import (
    prepare_smp_denoising_diagnostic_clip,
    select_playback_joint_names,
    select_playback_states,
    should_use_direct_joint_write,
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

    spec = importlib.util.spec_from_file_location("wbt_g1_cfg_smp_diagnostic", module_path)
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


def _save_debug_npz(path: str | Path, diagnostic_clip) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clean_frames": diagnostic_clip.clean_frames.detach().cpu().numpy(),
        "noisy_frames": diagnostic_clip.noisy_frames.detach().cpu().numpy(),
        "denoised_frames": diagnostic_clip.denoised_frames.detach().cpu().numpy(),
        "fps": np.asarray([diagnostic_clip.fps], dtype=np.float32),
        "frame_range": np.asarray(diagnostic_clip.frame_range, dtype=np.int64),
        "noise_step": np.asarray([diagnostic_clip.noise_step], dtype=np.int64),
        "raw_joint_names": np.asarray(diagnostic_clip.raw_joint_names, dtype=np.str_),
        "decoded_joint_names": np.asarray(diagnostic_clip.decoded_joint_names, dtype=np.str_),
    }
    if diagnostic_clip.source_name is not None:
        payload["source_name"] = np.asarray([diagnostic_clip.source_name], dtype=np.str_)
    for metric_name, metric_value in diagnostic_clip.metrics.items():
        payload[metric_name] = np.asarray([metric_value], dtype=np.float32)
    np.savez(output_path, **payload)
    return output_path


def _play_state_sequence(
    sim: SimulationContext,
    robot: Articulation,
    playback_states,
    playback_joint_names: tuple[str, ...],
    reference_joint_names: tuple[str, ...],
) -> None:
    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos[0].clone()
    default_joint_vel = robot.data.default_joint_vel[0].clone()
    robot_joint_names = tuple(str(name) for name in robot.data.joint_names)
    use_direct_joint_write = should_use_direct_joint_write(
        joint_state_width=int(playback_states.joint_pos.shape[1]),
        robot_joint_names=robot_joint_names,
        playback_joint_names=playback_joint_names,
        reference_joint_names=reference_joint_names,
        play_source=args_cli.play_source,
    )

    joint_ids_tensor = torch.tensor([], dtype=torch.long, device=sim.device)
    if not use_direct_joint_write:
        joint_ids, resolved_names = robot.find_joints(list(playback_joint_names), preserve_order=True)
        if tuple(resolved_names) != playback_joint_names:
            raise RuntimeError("Resolved robot joint order does not match playback joint names.")
        joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=sim.device)

    num_frames = int(playback_states.joint_pos.shape[0])
    while simulation_app.is_running() and not simulation_app.is_exiting():
        for frame_idx in range(num_frames):
            if not simulation_app.is_running() or simulation_app.is_exiting():
                return

            if use_direct_joint_write:
                full_joint_pos = playback_states.joint_pos[frame_idx]
                full_joint_vel = playback_states.joint_vel[frame_idx]
            else:
                full_joint_pos = default_joint_pos.clone()
                full_joint_vel = default_joint_vel.clone()
                full_joint_pos[joint_ids_tensor] = playback_states.joint_pos[frame_idx]
                full_joint_vel[joint_ids_tensor] = playback_states.joint_vel[frame_idx]

            root_state = torch.cat(
                (
                    playback_states.root_pos_w[frame_idx],
                    playback_states.root_quat_w[frame_idx],
                    playback_states.root_lin_vel_w[frame_idx],
                    playback_states.root_ang_vel_w[frame_idx],
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

            pos_lookat = playback_states.root_pos_w[frame_idx].detach().cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.6]), pos_lookat)

        if args_cli.play_once:
            return


def main() -> None:
    diagnostic_clip = prepare_smp_denoising_diagnostic_clip(
        dataset=args_cli.dataset,
        checkpoint=args_cli.checkpoint,
        style_id=args_cli.style_id,
        clip_start_sec=args_cli.clip_start_sec,
        clip_duration_sec=args_cli.clip_duration_sec,
        noise_step=args_cli.noise_step,
        device=args_cli.device,
        stride=args_cli.stride,
    )
    playback_states = select_playback_states(
        play_source=args_cli.play_source,
        raw_clip=diagnostic_clip.raw_clip,
        decoded_state_map=diagnostic_clip.decoded_state_map,
    )
    playback_joint_names = select_playback_joint_names(
        play_source=args_cli.play_source,
        raw_joint_names=diagnostic_clip.raw_joint_names,
        decoded_joint_names=diagnostic_clip.decoded_joint_names,
    )
    if args_cli.save_debug_npz is not None:
        saved_path = _save_debug_npz(args_cli.save_debug_npz, diagnostic_clip)
        print("[SMP Diagnostic] saved_debug_npz:", saved_path)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / diagnostic_clip.fps, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    robot = _design_scene()
    sim.reset()
    use_direct_joint_write = should_use_direct_joint_write(
        joint_state_width=int(playback_states.joint_pos.shape[1]),
        robot_joint_names=tuple(str(name) for name in robot.data.joint_names),
        playback_joint_names=playback_joint_names,
        reference_joint_names=diagnostic_clip.raw_joint_names,
        play_source=args_cli.play_source,
    )

    print("[SMP Diagnostic] dataset:", diagnostic_clip.dataset_path)
    print("[SMP Diagnostic] source_name:", diagnostic_clip.source_name)
    print("[SMP Diagnostic] fps:", f"{diagnostic_clip.fps:.3f}")
    print("[SMP Diagnostic] frame_range:", diagnostic_clip.frame_range)
    print("[SMP Diagnostic] num_frames:", int(diagnostic_clip.clean_frames.shape[0]))
    print("[SMP Diagnostic] noise_step:", diagnostic_clip.noise_step)
    print("[SMP Diagnostic] play_source:", args_cli.play_source)
    print("[SMP Diagnostic] clean_vs_noisy_mse:", f"{diagnostic_clip.metrics['clean_vs_noisy_mse']:.8f}")
    print("[SMP Diagnostic] clean_vs_denoised_mse:", f"{diagnostic_clip.metrics['clean_vs_denoised_mse']:.8f}")
    print("[SMP Diagnostic] joint_block_mse:", f"{diagnostic_clip.metrics['joint_block_mse']:.8f}")
    print("[SMP Diagnostic] velocity_block_mse:", f"{diagnostic_clip.metrics['velocity_block_mse']:.8f}")
    print(
        "[SMP Diagnostic] clean_decoded_vs_raw_joint_mae:",
        f"{diagnostic_clip.metrics['clean_decoded_vs_raw_joint_mae']:.8f}",
    )
    print(
        "[SMP Diagnostic] clean_decoded_vs_raw_joint_max_abs:",
        f"{diagnostic_clip.metrics['clean_decoded_vs_raw_joint_max_abs']:.8f}",
    )
    print(
        "[SMP Diagnostic] clean_decoded_vs_raw_root_pos_mae:",
        f"{diagnostic_clip.metrics['clean_decoded_vs_raw_root_pos_mae']:.8f}",
    )
    print(
        "[SMP Diagnostic] clean_decoded_vs_raw_root_quat_mae:",
        f"{diagnostic_clip.metrics['clean_decoded_vs_raw_root_quat_mae']:.8f}",
    )
    print("[SMP Diagnostic] joint_write:", "direct_full_vector" if use_direct_joint_write else "mapped_by_joint_names")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        _play_state_sequence(sim, robot, playback_states, playback_joint_names, diagnostic_clip.raw_joint_names)


if __name__ == "__main__":
    main()
    simulation_app.close()
