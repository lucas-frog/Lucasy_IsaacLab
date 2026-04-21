# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _import_trainer_module():
    repo_root = Path(__file__).resolve().parents[3]
    nested_rsl_rl_root = repo_root / "rsl_rl"
    if str(nested_rsl_rl_root) not in sys.path:
        sys.path.insert(0, str(nested_rsl_rl_root))
    for module_name in [name for name in sys.modules if name == "rsl_rl" or name.startswith("rsl_rl.")]:
        sys.modules.pop(module_name, None)
    return importlib.import_module("rsl_rl.diffusion.trainer")


def _write_toy_motion_dataset(path: Path, num_frames: int = 12, feature_dim: int = 6) -> None:
    frames = np.linspace(-1.0, 1.0, num_frames * feature_dim, dtype=np.float32).reshape(num_frames, feature_dim)
    np.savez(path, fps=np.array([30], dtype=np.int64), frames=frames)


def test_smp_trainer_pretrain_samples_all_diffusion_steps(tmp_path, monkeypatch):
    trainer_module = _import_trainer_module()
    dataset_path = tmp_path / "toy_motion.npz"
    _write_toy_motion_dataset(dataset_path)

    trainer = trainer_module.SMPDiffusionTrainer(
        dataset_path=dataset_path,
        log_dir=tmp_path / "logs",
        batch_size=4,
        max_iters=1,
        window_size=4,
        stride=1,
        num_diffusion_steps=7,
        timesteps_k=[5, 3, 1],
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        device="cpu",
    )

    observed = {}

    def _fake_sample_timesteps(batch_size: int, device=None, timesteps_k=None):
        observed["batch_size"] = batch_size
        observed["timesteps_k"] = timesteps_k
        return torch.tensor([0, 1, 2, 6], device=device, dtype=torch.long)

    monkeypatch.setattr(trainer.scheduler, "sample_timesteps", _fake_sample_timesteps)

    trainer._compute_loss(trainer._next_batch())

    assert observed["batch_size"] == 4
    assert observed["timesteps_k"] is None


def test_smp_trainer_prints_pretrain_progress(tmp_path, monkeypatch, capsys):
    trainer_module = _import_trainer_module()
    dataset_path = tmp_path / "toy_motion.npz"
    _write_toy_motion_dataset(dataset_path)
    monkeypatch.setattr(trainer_module, "log_smp_pretrain_metrics", lambda *args, **kwargs: None)

    trainer = trainer_module.SMPDiffusionTrainer(
        dataset_path=dataset_path,
        log_dir=tmp_path / "logs",
        batch_size=2,
        max_iters=2,
        window_size=4,
        stride=1,
        num_diffusion_steps=7,
        timesteps_k=[5, 3, 1],
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        device="cpu",
    )

    result = trainer.train()
    output = capsys.readouterr().out
    lines = output.splitlines()

    assert result["final_loss"] is not None
    assert "[SMP Pretrain] Starting" in output
    assert "Pretrain timesteps:" in output
    assert "Reward timesteps k:" in output
    assert any("Learning iteration 1/2" in line for line in lines)
    assert any("Learning iteration 2/2" in line for line in lines)
    assert any("Computation:" in line for line in lines)
    assert any("Mean loss:" in line for line in lines)
    assert any("Mean loss (running):" in line for line in lines)
    assert any("Estimated total time:" in line for line in lines)
    assert any("Time remaining:" in line for line in lines)
    assert any("Checkpoint:" in line for line in lines)


def test_smp_trainer_warns_when_style_labels_exist_but_style_drop_is_zero(tmp_path, monkeypatch):
    trainer_module = _import_trainer_module()
    dataset_path = tmp_path / "toy_motion_labeled.npz"
    frames = np.linspace(-1.0, 1.0, 12 * 6, dtype=np.float32).reshape(12, 6)
    np.savez(
        dataset_path,
        fps=np.array([30], dtype=np.int64),
        frames=frames,
        feature_schema=np.asarray(["extended_198"], dtype=np.str_),
        style_name=np.asarray(["walk"], dtype=np.str_),
        style_id=np.asarray([0], dtype=np.int64),
    )

    warnings_seen = []
    monkeypatch.setattr(
        trainer_module.warnings,
        "warn",
        lambda message, *args, **kwargs: warnings_seen.append(str(message)),
    )

    trainer_module.SMPDiffusionTrainer(
        dataset_path=dataset_path,
        log_dir=tmp_path / "logs",
        batch_size=2,
        max_iters=1,
        window_size=4,
        stride=1,
        num_diffusion_steps=7,
        timesteps_k=[5, 3, 1],
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        style_drop_prob=0.0,
        device="cpu",
    )
    assert any("style_drop_prob=0.0" in message for message in warnings_seen)
