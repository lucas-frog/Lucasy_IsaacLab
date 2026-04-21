# SMP Reward Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible pipeline that collects policy `smp_motion_window` rollouts, converts raw positive/negative motion datasets, scores all three groups with a frozen SMP prior, and prints key comparison metrics.

**Architecture:** Reuse the existing pure diagnostics module for window loading and prior scoring, keep Isaac-Sim-dependent rollout collection as a thin wrapper in `play.py`, and put offline comparison logic in a small standalone CLI. New code should expose pure helpers first so unit tests can cover dataset conversion, summary generation, and collection metadata without booting the simulator.

**Tech Stack:** Python, NumPy, PyTorch, pytest, IsaacLab RSL-RL integration

---

## File Map

- Modify `source/isaaclab_rl/test/test_smp_diagnostics.py` - keep diagnostics helpers testable and cover new summary helpers if needed.
- Create `source/isaaclab_rl/test/test_smp_reward_evaluation.py` - test offline comparison CLI logic with fake scoring helpers.
- Modify `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_diagnostics.py` - add any missing pure helpers for policy collection and report serialization.
- Modify `scripts/reinforcement_learning/rsl_rl/play.py` - resolve custom runner classes and optionally dump `smp_motion_window` policy rollouts.
- Create `scripts/imitation_learning/smp/evaluate_smp_reward.py` - convert raw positive/negative datasets, load policy windows, score all groups, print key metrics, and optionally save JSON.

### Task 1: Stabilize Diagnostics Module Tests

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_diagnostics.py`

- [x] **Step 1: Fix the dynamic import harness so dataclass annotations load correctly**
- [x] **Step 2: Run `pytest -q source/isaaclab_rl/test/test_smp_diagnostics.py` and confirm green**

### Task 2: Add Offline Evaluation Tests First

**Files:**
- Create: `source/isaaclab_rl/test/test_smp_reward_evaluation.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_diagnostics.py`
- Create: `scripts/imitation_learning/smp/evaluate_smp_reward.py`

- [ ] **Step 1: Write a failing test that verifies raw positive/negative data, policy windows, and summary metrics flow through one evaluation entrypoint**
- [ ] **Step 2: Run the new test and confirm it fails for missing script/helpers**
- [ ] **Step 3: Implement the minimal evaluation entrypoint and report formatting**
- [ ] **Step 4: Re-run the focused tests and confirm green**

### Task 3: Add Policy Window Dumping Support

**Files:**
- Modify: `scripts/reinforcement_learning/rsl_rl/play.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_diagnostics.py`

- [ ] **Step 1: Write a failing test for the pure helper that infers collector metadata from flattened `smp_motion_window` observations**
- [ ] **Step 2: Run the diagnostics tests and confirm the new helper test fails**
- [ ] **Step 3: Implement the helper and wire `play.py` to optionally collect and save rollout windows**
- [ ] **Step 4: Re-run the focused tests and confirm green**

### Task 4: Verify The End-to-End Diagnostic Surface

**Files:**
- Test: `source/isaaclab_rl/test/test_smp_diagnostics.py`
- Test: `source/isaaclab_rl/test/test_smp_reward_evaluation.py`

- [ ] **Step 1: Run the focused pytest targets**
- [ ] **Step 2: Review CLI help / output formatting for usability**
