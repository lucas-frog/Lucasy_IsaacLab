# SMP Denoised Motion Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI tool that loads a G1 SMP dataset clip, applies forward diffusion noise, reconstructs the denoised clip with a trained prior checkpoint, and plays only the denoised result in Isaac Sim.

**Architecture:** Keep Isaac-specific app/simulation lifecycle inside a new script under `scripts/imitation_learning/smp/`, and move deterministic data selection, windowing, denoising reconstruction, overlap stitching, and approximate playback-state decoding into a testable helper module under `source/isaaclab_rl/isaaclab_rl/rsl_rl/`. Reuse existing diffusion scheduler/model math and existing G1 feature layout constants rather than duplicating format definitions.

**Tech Stack:** Python, PyTorch, Isaac Lab `AppLauncher`/`SimulationContext`, `isaaclab_assets.G1_CFG`, `rsl_rl.diffusion` (`DiffusionScheduler`, `MotionEpsilonTransformer`), `pytest`

---

### Task 1: Add failing tests for dataset resolution and clip slicing

**Files:**
- Create: `source/isaaclab_rl/test/test_smp_denoised_playback.py`
- Reference: `docs/superpowers/specs/2026-04-15-smp-denoised-motion-playback-design.md`

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_dataset_npz_from_single_file(tmp_path):
    ...


def test_resolve_dataset_npz_from_corpus_dir_by_source_name(tmp_path):
    ...


def test_slice_clip_frames_uses_fps_and_duration():
    ...


def test_slice_clip_frames_rejects_out_of_range_request():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "resolve_dataset or slice_clip" -v`
Expected: FAIL with missing module/function errors

- [ ] **Step 3: Write minimal implementation**

Create helpers with exact responsibilities:

```python
def resolve_dataset_npz(dataset_path: str | Path, source_name: str | None = None) -> Path:
    ...


def load_smp_dataset_npz(path: str | Path) -> dict[str, object]:
    ...


def slice_clip_frames(
    frames: torch.Tensor,
    fps: float,
    clip_start_sec: float,
    clip_duration_sec: float,
) -> tuple[torch.Tensor, tuple[int, int]]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "resolve_dataset or slice_clip" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_rl/test/test_smp_denoised_playback.py source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py
git commit -m "test: add dataset resolution and clip slicing coverage"
```

### Task 2: Add failing tests for windowing, denoising reconstruction, and overlap stitching

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_denoised_playback.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_motion_windows_returns_expected_shape():
    ...


def test_stitch_motion_windows_averages_overlaps():
    ...


def test_reconstruct_x0_from_eps_matches_closed_form_solution():
    ...


def test_denoise_motion_windows_uses_model_style_id_when_available():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "window or stitch or reconstruct_x0 or denoise" -v`
Expected: FAIL with missing helper errors

- [ ] **Step 3: Write minimal implementation**

Implement exact helpers:

```python
def build_motion_windows(frames: torch.Tensor, window_size: int, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    ...


def reconstruct_x0_from_eps(
    xt: torch.Tensor,
    eps_hat: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    ...


def denoise_motion_windows(...):
    ...


def stitch_motion_windows(...):
    ...
```

Use `DiffusionScheduler.q_sample()` for forward noising and the DDPM closed-form `x0_hat` reconstruction.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "window or stitch or reconstruct_x0 or denoise" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_rl/test/test_smp_denoised_playback.py source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py
git commit -m "feat: add smp denoising window helpers"
```

### Task 3: Add failing tests for approximate playback state decoding

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_denoised_playback.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_decode_joint_positions_from_rot6d_recovers_default_pose_offsets():
    ...


def test_reconstruct_playback_states_integrates_heading_and_position():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "decode_joint_positions or reconstruct_playback_states" -v`
Expected: FAIL with missing helper errors

- [ ] **Step 3: Write minimal implementation**

Implement decoding helpers that reuse existing G1 layout assumptions:

```python
@dataclass
class PlaybackStateSequence:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


def decode_joint_positions_from_rot6d(...):
    ...


def reconstruct_playback_states(...):
    ...
```

Use default G1 root height, yaw-only heading integration, world-frame velocity reconstruction, and finite differences for joint velocities.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "decode_joint_positions or reconstruct_playback_states" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_rl/test/test_smp_denoised_playback.py source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py
git commit -m "feat: add playback state reconstruction helpers"
```

### Task 4: Add failing tests for checkpoint loading and end-to-end clip preparation

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_denoised_playback.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_smp_prior_checkpoint_restores_model_cfg_and_style_id(tmp_path):
    ...


def test_prepare_denoised_motion_clip_returns_metrics_and_playback_states(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "load_smp_prior_checkpoint or prepare_denoised_motion_clip" -v`
Expected: FAIL with missing exports/helpers

- [ ] **Step 3: Write minimal implementation**

Add exact high-level helpers:

```python
@dataclass
class LoadedSmpPrior:
    model: MotionEpsilonTransformer
    scheduler: DiffusionScheduler
    style_id: int | None
    window_size: int
    feature_dim: int
    num_diffusion_steps: int


def load_smp_prior_checkpoint(...):
    ...


def prepare_denoised_motion_clip(...):
    ...
```

Export the public helpers from `source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -k "load_smp_prior_checkpoint or prepare_denoised_motion_clip" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_rl/test/test_smp_denoised_playback.py source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py
git commit -m "feat: add smp denoised clip preparation api"
```

### Task 5: Add the Isaac playback CLI script

**Files:**
- Create: `scripts/imitation_learning/smp/play_denoised_motion_clip.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py`
- Reference: `scripts/demos/bipeds.py`

- [ ] **Step 1: Write the failing test or verification target**

Since Isaac AppLauncher scripts are not practical for `pytest`, define a smoke-level verification target instead:

```bash
python scripts/imitation_learning/smp/play_denoised_motion_clip.py --help
```

Expected behavior: CLI renders all required arguments without import errors.

- [ ] **Step 2: Run verification to confirm it currently fails**

Run: `python scripts/imitation_learning/smp/play_denoised_motion_clip.py --help`
Expected: FAIL because the script does not exist

- [ ] **Step 3: Write minimal implementation**

Implement the script with these sections:

```python
parser.add_argument("--dataset", required=True, ...)
parser.add_argument("--checkpoint", required=True, ...)
parser.add_argument("--source-name", default=None, ...)
parser.add_argument("--clip-start-sec", type=float, default=5.0, ...)
parser.add_argument("--clip-duration-sec", type=float, default=5.0, ...)
parser.add_argument("--noise-step", type=int, default=15, ...)
parser.add_argument("--save-debug-npz", default=None, ...)
```

Flow:
1. Launch Isaac app
2. Build simulation scene with `G1_CFG`
3. Call `prepare_denoised_motion_clip(...)`
4. Print metrics
5. Reset robot to first frame
6. Iterate through reconstructed playback states and write them to sim at dataset FPS

- [ ] **Step 4: Run verification to confirm it passes**

Run: `python scripts/imitation_learning/smp/play_denoised_motion_clip.py --help`
Expected: PASS and prints CLI usage

- [ ] **Step 5: Commit**

```bash
git add scripts/imitation_learning/smp/play_denoised_motion_clip.py source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py
git commit -m "feat: add smp denoised motion playback script"
```

### Task 6: Validate targeted tests and document usage

**Files:**
- Modify: `docs/superpowers/specs/2026-04-15-smp-denoised-motion-playback-design.md`
- Reference: `scripts/imitation_learning/smp/play_denoised_motion_clip.py`

- [ ] **Step 1: Run the focused unit tests**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -v`
Expected: PASS

- [ ] **Step 2: Run the CLI help smoke check**

Run: `python scripts/imitation_learning/smp/play_denoised_motion_clip.py --help`
Expected: PASS

- [ ] **Step 3: Add a usage note to the design/spec document**

Append a short “Run command” example such as:

```bash
./isaaclab.sh -p scripts/imitation_learning/smp/play_denoised_motion_clip.py \
  --dataset /home/lucas/whole_body_tracking/artifacts/combine/g1_walk_corpus \
  --source-name walk1 \
  --checkpoint /home/lucas/isaac-sim/IsaacLab/logs/smp_prior/g1/pretrain_407/model_latest.pt \
  --clip-start-sec 5.0 \
  --clip-duration-sec 5.0 \
  --noise-step 15
```

- [ ] **Step 4: Re-run the focused unit tests if docs touched code-adjacent examples**

Run: `pytest source/isaaclab_rl/test/test_smp_denoised_playback.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-15-smp-denoised-motion-playback-design.md source/isaaclab_rl/test/test_smp_denoised_playback.py scripts/imitation_learning/smp/play_denoised_motion_clip.py source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py
git commit -m "docs: add smp denoised playback usage example"
```
