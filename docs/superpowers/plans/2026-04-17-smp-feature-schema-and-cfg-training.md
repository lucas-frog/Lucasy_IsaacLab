# SMP Feature Schema And CFG Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a schema-selectable SMP feature pipeline that keeps existing `192D` datasets/checkpoints/configs working, introduces an opt-in `198D` format with appended `base_lin_vel_w` and `base_ang_vel_w`, preserves paired-dataset replay, and trains the single style-conditioned diffusion prior in a paper-aligned CFG setup via style dropout.

**Architecture:** Centralize schema-aware SMP feature layout in the shared offline/online feature helpers, then thread `feature_schema` through paired dataset metadata, CSV export, replay, RL observation config, diffusion dataset/trainer, and the SMP runner. Keep the diffusion model architecture unchanged: one shared style-conditioned model learns both conditional and null-style predictions through `style_drop_prob`, while runtime CFG continues to compute `f(x_i, ∅)` and `f(x_i, c)` from the same frozen network.

**Tech Stack:** Python, NumPy, PyTorch, Isaac Lab `AppLauncher`, existing SMP paired-dataset helpers, G1 locomotion config, pytest.

---

## File Map

- Modify `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_paired_dataset.py` - add schema-aware feature layout helpers, offline `198D` packing, and paired-dataset metadata support.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py` - add matching online schema-aware frame packing for RL observations.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py` - plumb `feature_schema` into `smp_frame_features(...)`.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py` - expose G1 SMP feature dims and block offsets by schema instead of one hard-coded `192D` layout.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` - let the `smp_motion_window` observation select schema explicitly.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py` - pass the chosen schema, `feature_dim`, and block offsets into `SMPPriorCfg` and `SMPStyleCfg`.
- Modify `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py` - add `feature_schema` to `SMPPriorCfg`.
- Modify `scripts/imitation_learning/smp/csv_to_smp_paired_dataset.py` - add `--feature-schema` and emit schema-aware `frames`.
- Modify `scripts/imitation_learning/smp/replay_smp_paired_dataset.py` - keep replay raw-state-based and make any frame-related checks schema-aware.
- Modify `rsl_rl/rsl_rl/motion/smp_dataset.py` - read and expose `feature_schema` metadata without hard-coding `192`.
- Modify `rsl_rl/rsl_rl/diffusion/trainer.py` - propagate dataset schema metadata, keep single-model style-drop training, and warn clearly when style labels exist but `style_drop_prob == 0.0`.
- Modify `scripts/imitation_learning/smp/train_motion_prior.py` - surface the CFG-training intent in CLI help and warnings.
- Modify `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py` - validate runtime window dims against prior config/checkpoint dims and keep CFG inference schema-agnostic.
- Modify `source/isaaclab_rl/test/test_smp_feature_utils.py` - cover `192D` and `198D` packing.
- Modify `source/isaaclab_rl/test/test_smp_paired_dataset.py` - cover paired metadata and `198D` raw-motion export.
- Modify `source/isaaclab_rl/test/test_smp_dataset.py` - cover loading `feature_schema` from dataset NPZs.
- Modify `source/isaaclab_rl/test/test_smp_trainer.py` - cover `198D` training data and CFG-drop warnings.
- Modify `source/isaaclab_rl/test/test_smp_runner_style_program.py` - cover runner window restoration / mismatch errors with non-`192D` priors.

### Task 1: Add schema-aware shared feature packing

**Files:**
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_paired_dataset.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`
- Test: `source/isaaclab_rl/test/test_smp_feature_utils.py`
- Test: `source/isaaclab_rl/test/test_smp_paired_dataset.py`

- [ ] **Step 1: Write the failing tests**

Add focused tests that expect:

- `pack_smp_frame_features(...)` to keep `legacy_192` unchanged
- `pack_smp_frame_features(...)` to emit `198` dims for `extended_198`
- the last `6` dims of `extended_198` to be `[base_lin_vel_w, base_ang_vel_w]`
- paired-dataset block metadata for `extended_198` to include `base_lin_vel_w` and `base_ang_vel_w`
- `build_g1_smp_frames_from_raw_motion(...)` to accept `expected_feature_dim=21` in a toy one-joint/one-EE case when world velocities are appended

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest source/isaaclab_rl/test/test_smp_feature_utils.py source/isaaclab_rl/test/test_smp_paired_dataset.py -q`

Expected: FAIL because the packers and paired-dataset helpers currently only understand the existing `192D` layout.

- [ ] **Step 3: Implement minimal schema helpers**

Introduce a small shared schema surface in both feature helper modules:

```python
LEGACY_SMP_FEATURE_SCHEMA = "legacy_192"
EXTENDED_SMP_FEATURE_SCHEMA = "extended_198"

def g1_smp_feature_block_offsets(..., feature_schema: str = LEGACY_SMP_FEATURE_SCHEMA) -> dict[str, tuple[int, int]]:
    ...

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
    ...
```

Behavior to preserve:

- `legacy_192` output order stays byte-for-byte compatible with the current layout
- `extended_198` appends world linear/angular velocity and nothing else changes
- offline and online helpers use the same schema names and append order

- [ ] **Step 4: Re-run focused tests**

Run: `pytest source/isaaclab_rl/test/test_smp_feature_utils.py source/isaaclab_rl/test/test_smp_paired_dataset.py -q`

Expected: PASS

### Task 2: Add paired-dataset metadata and schema-selectable CSV export

**Files:**
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_paired_dataset.py`
- Modify: `scripts/imitation_learning/smp/csv_to_smp_paired_dataset.py`
- Test: `source/isaaclab_rl/test/test_smp_paired_dataset.py`

- [ ] **Step 1: Extend failing paired-dataset tests to require `feature_schema`**

Update the paired-dataset save/load tests so they require:

- `feature_schema` to be written into the NPZ payload
- `feature_dim` and `feature_block_offsets` to match the chosen schema
- legacy payloads with no explicit `feature_schema` to remain readable by deriving `legacy_192` when `feature_dim == 192`

- [ ] **Step 2: Run the paired-dataset tests and verify failure**

Run: `pytest source/isaaclab_rl/test/test_smp_paired_dataset.py -q`

Expected: FAIL because `save_smp_paired_dataset(...)` does not currently write `feature_schema`.

- [ ] **Step 3: Implement schema metadata in the shared save/load path**

Update `save_smp_paired_dataset(...)` and the paired-dataset readers to:

- accept a `feature_schema` argument
- save `feature_schema` as a string field
- keep `feature_block_names`, `feature_block_offsets`, and `feature_dim` authoritative
- provide a small reader helper that returns a normalized schema for both new and legacy payloads

Keep the raw-state fields unchanged so replay remains independent of frame layout.

- [ ] **Step 4: Add `--feature-schema` to the CSV exporter**

Update `csv_to_smp_paired_dataset.py` so it:

- accepts `--feature-schema {legacy_192,extended_198}`
- uses the shared offline packer to build `frames`
- writes `feature_schema`, `feature_dim`, and schema-appropriate block offsets into the paired NPZ

Do not change the raw-state capture/write path.

- [ ] **Step 5: Re-run the paired-dataset tests**

Run: `pytest source/isaaclab_rl/test/test_smp_paired_dataset.py -q`

Expected: PASS

### Task 3: Keep replay compatible with new paired datasets

**Files:**
- Modify: `scripts/imitation_learning/smp/replay_smp_paired_dataset.py`
- Test: `source/isaaclab_rl/test/test_smp_paired_dataset.py`

- [ ] **Step 1: Write a regression test for new-schema payload loading**

Add a test that writes an `extended_198` paired dataset and verifies:

- `load_smp_paired_dataset(...)` can still read it
- `load_paired_raw_state_clip(...)` still slices raw states correctly
- replay-relevant helpers do not care whether `frames.shape[-1]` is `192` or `198`

- [ ] **Step 2: Run the focused replay-related tests**

Run: `pytest source/isaaclab_rl/test/test_smp_paired_dataset.py -q`

Expected: FAIL if any replay helper still assumes `192D`.

- [ ] **Step 3: Make replay metadata checks schema-aware**

Update `replay_smp_paired_dataset.py` only where needed so that:

- raw-state replay remains the primary playback path
- any informational output or validation that touches `frames`, `feature_dim`, or block offsets reads schema-aware metadata
- no part of the script rejects a paired dataset solely because its `frames` are `198D`

- [ ] **Step 4: Re-run the paired-dataset tests**

Run: `pytest source/isaaclab_rl/test/test_smp_paired_dataset.py -q`

Expected: PASS

### Task 4: Plumb schema into online RL SMP observations and configs

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py`
- Test: `source/isaaclab_rl/test/test_smp_feature_utils.py`
- Test: `source/isaaclab_rl/test/test_smp_runner_style_program.py`

- [ ] **Step 1: Write the failing tests/config assertions**

Add or extend tests so they require:

- G1 config helpers to compute `feature_dim` and block offsets from a schema choice
- runner/window restoration logic to accept any configured `feature_dim`, not just `192`

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest source/isaaclab_rl/test/test_smp_feature_utils.py source/isaaclab_rl/test/test_smp_runner_style_program.py -q`

Expected: FAIL because the config and runner plumbing still assume a single fixed G1 SMP layout.

- [ ] **Step 3: Add schema-aware G1 config helpers**

Refactor the G1 agent config module so it exports:

```python
g1_smp_feature_schema = "legacy_192"

def g1_smp_feature_dim_for_schema(feature_schema: str) -> int:
    ...

def g1_smp_feature_block_offsets_for_schema(feature_schema: str) -> dict[str, tuple[int, int]]:
    ...
```

Then update:

- `SMPPriorCfg` to carry `feature_schema`
- `velocity_env_cfg.py` to pass `feature_schema` into `smp_frame_features(...)`
- `rsl_rl_ppo_cfg.py` to use schema-derived dims and block offsets

- [ ] **Step 4: Extend the observation term**

Update `smp_frame_features(...)` so:

- `feature_schema="legacy_192"` returns the current window frame unchanged
- `feature_schema="extended_198"` appends `asset.data.root_lin_vel_w` and `asset.data.root_ang_vel_w` after the existing `192D` feature vector

- [ ] **Step 5: Re-run focused tests**

Run: `pytest source/isaaclab_rl/test/test_smp_feature_utils.py source/isaaclab_rl/test/test_smp_runner_style_program.py -q`

Expected: PASS

### Task 5: Make dataset loading and prior training schema-aware while keeping single-model CFG training

**Files:**
- Modify: `rsl_rl/rsl_rl/motion/smp_dataset.py`
- Modify: `rsl_rl/rsl_rl/diffusion/trainer.py`
- Modify: `scripts/imitation_learning/smp/train_motion_prior.py`
- Test: `source/isaaclab_rl/test/test_smp_dataset.py`
- Test: `source/isaaclab_rl/test/test_smp_trainer.py`

- [ ] **Step 1: Write failing dataset/trainer tests**

Add tests that expect:

- `SMPMotionWindowDataset` to expose `feature_schema` when the NPZ contains it
- a `198D` NPZ to load and yield windows without special cases
- trainer initialization on a style-labeled dataset with `style_drop_prob == 0.0` to emit a clear warning

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest source/isaaclab_rl/test/test_smp_dataset.py source/isaaclab_rl/test/test_smp_trainer.py -q`

Expected: FAIL because the dataset does not currently retain `feature_schema`, and the trainer does not surface a CFG-training warning.

- [ ] **Step 3: Implement dataset schema propagation**

Update `SMPMotionWindowDataset` to:

- read `feature_schema` when present
- infer `legacy_192` for old `192D` payloads without schema
- store the normalized schema on the dataset object for logging/debugging

Do not reject non-`192D` frames as long as the file is a valid 2D `(num_frames, feature_dim)` array.

- [ ] **Step 4: Keep trainer paper-aligned and add warning semantics**

Keep the existing single-forward training structure:

- `style_id -> maybe_drop_style(...) -> self.model(...)`

Only add:

- clear log/warning text when style labels are present but `style_drop_prob == 0.0`
- optional checkpoint metadata for `feature_schema` if it is not already recoverable from the dataset

Do not add a second model or a dual-loss double-forward training path.

- [ ] **Step 5: Update CLI help text**

Adjust `train_motion_prior.py` help/summary text so it explains that non-zero `style_drop_prob` is the intended way to train null-style predictions for CFG.

- [ ] **Step 6: Re-run focused tests**

Run: `pytest source/isaaclab_rl/test/test_smp_dataset.py source/isaaclab_rl/test/test_smp_trainer.py -q`

Expected: PASS

### Task 6: Validate the runtime prior/runner path for mixed `192D` and `198D` setups

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- Modify: `source/isaaclab_rl/test/test_smp_runner_style_program.py`

- [ ] **Step 1: Write failing runner tests**

Add focused tests that expect:

- `_restore_smp_window(...)` to reshape according to `window_size * feature_dim` from config/checkpoint, even when `feature_dim != 192`
- a mismatched env/prior flattened dim to raise a clear error showing both expected and actual sizes

- [ ] **Step 2: Run the focused runner tests**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py -q`

Expected: FAIL because the new schema-aware config path is not yet fully wired through the runner.

- [ ] **Step 3: Implement runtime validation**

Update `SMPOnPolicyRunner` so it:

- preserves the current CFG inference logic
- trusts `model_cfg["feature_dim"]` / `SMPPriorCfg.feature_dim` instead of any hidden `192D` assumption
- optionally stores/reads `feature_schema` for diagnostics
- raises early when the restored observation window shape does not match `window_size * feature_dim`

- [ ] **Step 4: Re-run the focused runner tests**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py -q`

Expected: PASS

### Task 7: Targeted verification and operator commands

**Files:**
- Test: `source/isaaclab_rl/test/test_smp_feature_utils.py`
- Test: `source/isaaclab_rl/test/test_smp_paired_dataset.py`
- Test: `source/isaaclab_rl/test/test_smp_dataset.py`
- Test: `source/isaaclab_rl/test/test_smp_trainer.py`
- Test: `source/isaaclab_rl/test/test_smp_runner_style_program.py`

- [ ] **Step 1: Run the focused pytest suite**

Run:

```bash
pytest \
  source/isaaclab_rl/test/test_smp_feature_utils.py \
  source/isaaclab_rl/test/test_smp_paired_dataset.py \
  source/isaaclab_rl/test/test_smp_dataset.py \
  source/isaaclab_rl/test/test_smp_trainer.py \
  source/isaaclab_rl/test/test_smp_runner_style_program.py -q
```

Expected: PASS

- [ ] **Step 2: Check both script help entry points**

Run:

```bash
python scripts/imitation_learning/smp/csv_to_smp_paired_dataset.py --help
python scripts/imitation_learning/smp/replay_smp_paired_dataset.py --help
```

Expected: both commands print usage successfully without requiring a live Isaac Sim runtime.

- [ ] **Step 3: Smoke-test the new export/replay workflow**

Run an operator-level sequence after implementation:

```bash
python scripts/imitation_learning/smp/csv_to_smp_paired_dataset.py \
  --input-file /path/to/motion.csv \
  --output /tmp/smp_walk_198.npz \
  --feature-schema extended_198 \
  --window-size 10 \
  --headless

python scripts/imitation_learning/smp/replay_smp_paired_dataset.py \
  --dataset /tmp/smp_walk_198.npz \
  --play-once
```

Expected: export completes, the output NPZ contains `feature_schema=extended_198`, and replay runs from raw states without rejecting the `198D` `frames`.
