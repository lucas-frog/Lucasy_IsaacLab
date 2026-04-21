# SMP Reward Tag Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename SMP reward breakdown TensorBoard tags to avoid the `SMP/reward` prefix collision and make them easier to find in TensorBoard.

**Architecture:** Keep the change local to `SMPOnPolicyRunner` logging and its focused unit test. Preserve the existing scalar values and console output, only moving the breakdown tags from `SMP/reward/...` to `SMP/reward_terms/...`.

**Tech Stack:** Python, PyTorch, pytest, TensorBoard scalar logging

---

### Task 1: Update failing test expectations

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_runner_style_program.py`

- [ ] **Step 1: Write the failing test**

Change the log test to expect:
- `SMP/reward_terms/task_raw`
- `SMP/reward_terms/task_scaled`
- `SMP/reward_terms/style_raw`
- `SMP/reward_terms/style_scaled`
- `SMP/reward_terms/combined`

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py -k smp_log_reports_reward_breakdown -q`
Expected: FAIL because the runner still writes the old `SMP/reward/...` breakdown tags.

### Task 2: Rename runner tags

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`

- [ ] **Step 1: Rename TensorBoard tags**

Update the five breakdown scalars to use the `SMP/reward_terms/...` prefix while keeping:
- `SMP/reward`
- console diagnostics
- reward math

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py -k smp_log_reports_reward_breakdown -q`
Expected: PASS

### Task 3: Verify related SMP tests

**Files:**
- Test: `source/isaaclab_rl/test/test_smp_runner_style_program.py`
- Test: `source/isaaclab_rl/test/test_smp_reward_logging.py`

- [ ] **Step 1: Run targeted regression tests**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: PASS
