# SMP Heading-Local Feature Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate G1 SMP features from root-local velocity + joint scalar offsets + key-body rot6d to heading-local root velocity + per-joint relative rot6d + heading-local end-effector positions, using shared reusable transforms.

**Architecture:** Move the reusable coordinate-frame and joint-rotation encoding logic into the shared SMP feature utility module, then make both offline dataset export and online environment observation extraction call the same helpers. Update the G1 feature layout and downstream mask/GSI decoding utilities to understand the new joint rot6d block without breaking the SMP training window format.

**Tech Stack:** Python, PyTorch, IsaacLab task configs, rsl_rl diffusion utilities, pytest.

---

### Task 1: Lock the new feature contract with tests

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_feature_utils.py`
- Modify: `source/isaaclab_rl/test/test_smp_dataset.py`
- Modify: `source/isaaclab_rl/test/test_smp_gsi.py`

- [ ] Add failing tests for heading-local vector transforms and joint relative rot6d encoding
- [ ] Add failing dataset-export assertions for `192`-dim features and removed key-body block assumptions
- [ ] Add failing GSI layout/decode tests for `joint_rot6d_rel` support
- [ ] Run the targeted pytest commands and confirm expected failures

### Task 2: Implement shared SMP feature helpers

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`

- [ ] Add pure-torch helpers for heading-local frame construction and world-to-heading vector transforms
- [ ] Add pure-torch helpers for single-axis joint angle <-> quaternion/rot6d conversions relative to default pose
- [ ] Update `pack_smp_frame_features` to use `joint_rot6d_rel` and remove `key_body_quat_b`
- [ ] Run focused feature utility tests and confirm they pass

### Task 3: Switch export and online observation paths to shared helpers

**Files:**
- Modify: `scripts/imitation_learning/smp/export_g1_motion_dataset.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`

- [ ] Replace duplicated quaternion/frame math in the exporter with shared helper calls
- [ ] Replace online observation extraction with shared helper calls
- [ ] Update G1 feature dim/block offsets/joint-axis metadata to the new `192`-dim layout
- [ ] Run dataset and config-adjacent tests and confirm they pass

### Task 4: Update downstream layout consumers

**Files:**
- Modify: `rsl_rl/rsl_rl/diffusion/composition.py`
- Modify: `rsl_rl/rsl_rl/diffusion/gsi.py`
- Modify: any directly affected tests under `source/isaaclab_rl/test/`

- [ ] Teach composition masks to work with per-joint block widths for either scalar or rot6d joint blocks
- [ ] Teach GSI layout parsing and decode/re-encode logic to support `joint_rot6d_rel`
- [ ] Update affected tests to the new layout semantics
- [ ] Run targeted downstream pytest commands and confirm they pass

### Task 5: Verify the migration end-to-end

**Files:**
- No new source files expected

- [ ] Run the full targeted pytest subset covering feature utils, dataset export, GSI, and style composition
- [ ] Inspect diffs for accidental unrelated changes
- [ ] Summarize the migration and any remaining caveats for the user
