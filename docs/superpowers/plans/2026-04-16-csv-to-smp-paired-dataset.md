# CSV To SMP Paired Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CSV-to-NPZ export path that writes SMP training frames and raw replay states into one paired dataset, plus a replay script that defaults to raw-state playback.

**Architecture:** Put testable dataset validation and save helpers in `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_paired_dataset.py`. Add Isaac Sim CLI scripts under `scripts/imitation_learning/smp/`: one exporter that mirrors `whole_body_tracking/scripts/csv_to_npz.py` and adds `frames`, and one replay script that reads the paired NPZ and writes saved raw states to the G1 robot.

**Tech Stack:** Python, NumPy, PyTorch, Isaac Lab `AppLauncher`, existing G1 SMP feature helpers, pytest.

---

### Task 1: Paired Dataset Core

**Files:**
- Create: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_paired_dataset.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py`
- Test: `source/isaaclab_rl/test/test_smp_paired_dataset.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run focused tests and verify import failures**
- [ ] **Step 3: Implement metadata extraction, saving, validation, and raw-state loading**
- [ ] **Step 4: Run focused tests and verify pass**

### Task 2: CSV Export Script

**Files:**
- Create: `scripts/imitation_learning/smp/csv_to_smp_paired_dataset.py`

- [ ] **Step 1: Reuse the existing CSV interpolation and Isaac render-only state capture pattern**
- [ ] **Step 2: Compute SMP `frames` from captured raw states using shared feature helpers**
- [ ] **Step 3: Save paired dataset and print validation metrics**

### Task 3: Raw Replay Script

**Files:**
- Create: `scripts/imitation_learning/smp/replay_smp_paired_dataset.py`

- [ ] **Step 1: Load paired dataset raw states**
- [ ] **Step 2: Resolve WBT G1 robot config and write root/joint state each frame**
- [ ] **Step 3: Add `--clip-start-sec`, `--clip-duration-sec`, `--play-once`, and `--render-only`**

### Task 4: Verification

**Files:**
- Test: `source/isaaclab_rl/test/test_smp_paired_dataset.py`

- [ ] **Step 1: Run focused pytest**
- [ ] **Step 2: Check CLI help for both scripts if Isaac imports allow it**
- [ ] **Step 3: Provide example commands for export, replay, and old-dataset comparison**
