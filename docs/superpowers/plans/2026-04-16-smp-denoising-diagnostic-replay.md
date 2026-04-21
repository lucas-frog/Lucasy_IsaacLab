# SMP Denoising Diagnostic Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Isaac Sim diagnostic replay tool that loads paired SMP datasets, applies forward diffusion noise plus checkpoint denoising on 192D frames, and replays `raw`, `clean_decoded`, `noisy_decoded`, or `denoised_decoded` clips on a single robot.

**Architecture:** Keep raw replay on the already-fixed paired replay path, and reuse existing denoising helpers for window slicing, `q_sample`, `x0_hat` reconstruction, and decoded playback state recovery. Add a thin script-specific orchestration layer plus focused unit tests that prove play-source routing and denoising clip preparation work without needing Isaac Sim in test runs.

**Tech Stack:** Python, NumPy, PyTorch, Isaac Lab `AppLauncher`, existing `smp_paired_dataset.py`, existing `smp_denoised_playback.py`, pytest.

---

### Task 1: Add Failing Diagnostic Replay Tests

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_denoised_playback.py`
- Create: `source/isaaclab_rl/test/test_smp_denoising_diagnostic_replay.py`

- [ ] **Step 1: Write the failing test for play-source selection**

```python
def test_select_playback_states_uses_raw_clip_for_raw_mode():
    clip = types.SimpleNamespace(playback_states="raw-state")
    assert module.select_playback_states(play_source="raw", raw_clip=clip, decoded_state_map={}) == "raw-state"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_denoising_diagnostic_replay.py::test_select_playback_states_uses_raw_clip_for_raw_mode -q`
Expected: FAIL with missing function/module error.

- [ ] **Step 3: Write the failing test for denoised clip preparation**

```python
def test_prepare_diagnostic_clip_returns_clean_noisy_and_denoised_frames(tmp_path):
    ...
    assert result.clean_frames.shape == (6, 12)
    assert result.noisy_frames.shape == (6, 12)
    assert result.denoised_frames.shape == (6, 12)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_denoising_diagnostic_replay.py::test_prepare_diagnostic_clip_returns_clean_noisy_and_denoised_frames -q`
Expected: FAIL with missing function/module error.

### Task 2: Implement Diagnostic Replay Helpers

**Files:**
- Create: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoising_diagnostic.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py`

- [ ] **Step 1: Implement clip dataclass and play-source selector**

```python
@dataclass
class SMPDenoisingDiagnosticClip:
    raw_clip: PairedRawStateClip
    clean_frames: torch.Tensor
    noisy_frames: torch.Tensor
    denoised_frames: torch.Tensor
    decoded_state_map: dict[str, PlaybackStateSequence]
    metrics: dict[str, float]
```

- [ ] **Step 2: Implement paired-dataset denoising preparation**

```python
def prepare_smp_denoising_diagnostic_clip(...):
    payload = load_smp_paired_dataset(dataset)
    clean_frames, frame_range = slice_clip_frames(...)
    windows, starts = build_motion_windows(...)
    noisy_windows = scheduler.q_sample(...)
    denoised_windows = denoise_motion_windows(...)
    ...
```

- [ ] **Step 3: Implement decoded-state reconstruction map**

```python
decoded_state_map = {
    "clean_decoded": reconstruct_playback_states(...),
    "noisy_decoded": reconstruct_playback_states(...),
    "denoised_decoded": reconstruct_playback_states(...),
}
```

- [ ] **Step 4: Export helper symbols from package init**

```python
from .smp_denoising_diagnostic import (
    SMPDenoisingDiagnosticClip,
    prepare_smp_denoising_diagnostic_clip,
    select_playback_states,
)
```

- [ ] **Step 5: Run focused tests to verify pass**

Run: `pytest source/isaaclab_rl/test/test_smp_denoising_diagnostic_replay.py -q`
Expected: PASS

### Task 3: Implement Isaac Sim Replay Script

**Files:**
- Create: `scripts/imitation_learning/smp/replay_smp_denoising_diagnostic.py`

- [ ] **Step 1: Add CLI arguments**

```python
parser.add_argument("--dataset", ...)
parser.add_argument("--checkpoint", ...)
parser.add_argument("--clip-start-sec", ...)
parser.add_argument("--clip-duration-sec", ...)
parser.add_argument("--noise-step", ...)
parser.add_argument("--play-source", choices=("raw", "clean_decoded", "noisy_decoded", "denoised_decoded"))
parser.add_argument("--stride", ...)
parser.add_argument("--style-id", ...)
parser.add_argument("--save-debug-npz", ...)
```

- [ ] **Step 2: Reuse the stable WBT robot loading and raw replay write policy**

```python
robot = _design_scene()
diagnostic_clip = prepare_smp_denoising_diagnostic_clip(...)
playback_states = select_playback_states(...)
_play_state_sequence(sim, robot, playback_states, ...)
```

- [ ] **Step 3: Print numeric diagnostics before playback**

```python
print("[SMP Diagnostic] clean_vs_noisy_mse:", ...)
print("[SMP Diagnostic] clean_vs_denoised_mse:", ...)
```

- [ ] **Step 4: Optionally save debug npz**

```python
np.savez(path, clean_frames=..., noisy_frames=..., denoised_frames=..., ...)
```

### Task 4: Verification

**Files:**
- Test: `source/isaaclab_rl/test/test_smp_denoising_diagnostic_replay.py`
- Create: `scripts/imitation_learning/smp/replay_smp_denoising_diagnostic.py`

- [ ] **Step 1: Run focused pytest**

Run: `pytest source/isaaclab_rl/test/test_smp_denoising_diagnostic_replay.py -q`
Expected: PASS

- [ ] **Step 2: Run syntax validation**

Run: `python -m py_compile source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoising_diagnostic.py scripts/imitation_learning/smp/replay_smp_denoising_diagnostic.py`
Expected: exit 0

- [ ] **Step 3: Validate CLI help**

Run: `python scripts/imitation_learning/smp/replay_smp_denoising_diagnostic.py --help`
Expected: help text includes dataset/checkpoint/noise-step/play-source.

- [ ] **Step 4: Provide manual Isaac command**

```bash
./isaaclab.sh -p scripts/imitation_learning/smp/replay_smp_denoising_diagnostic.py \
  --dataset <paired_npz> \
  --checkpoint /home/lucas/isaac-sim/IsaacLab/logs/smp_prior/g1/pretrain_407/model_latest.pt \
  --clip-start-sec 5.0 \
  --clip-duration-sec 1.0 \
  --noise-step 15 \
  --play-source denoised_decoded \
  --play-once
```
