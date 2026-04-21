# SMP Reward Diagnostic Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit SMP reward diagnostics to TensorBoard and console output so training can show task reward, raw SMP reward, scaled SMP contribution, and combined reward.

**Architecture:** Keep the change localized to `SMPOnPolicyRunner`. Aggregate reward-term means during rollout, pass them through the existing logging payload, and emit both TensorBoard scalars and a compact console block. Add a focused unit test that exercises the runner log method without requiring a full training stack.

**Tech Stack:** Python, PyTorch, pytest, TensorBoard-style scalar logging

---

### Task 1: Add failing log test

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_runner_style_program.py`

- [ ] **Step 1: Write the failing test**

Add a unit test that constructs a lightweight `SMPOnPolicyRunner`, calls `log()`, and expects:
- TensorBoard scalar tags for explicit reward diagnostics
- Console output lines for the same reward diagnostics

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py -k smp_log_reports_reward_breakdown -q`
Expected: FAIL because the runner does not yet emit the new scalars or console text.

### Task 2: Implement reward diagnostic aggregation and logging

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`

- [ ] **Step 1: Add reward-term aggregation**

Introduce a small helper that computes:
- raw task reward
- scaled task reward
- raw SMP reward
- scaled SMP reward contribution
- combined reward

Aggregate their rollout means in the training loop.

- [ ] **Step 2: Emit logs**

Write the aggregated values into the `locs` payload and log:
- TensorBoard scalars under `SMP/reward/...`
- a compact console summary block after the base PPO log

- [ ] **Step 3: Run tests to verify pass**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py -k smp_log_reports_reward_breakdown -q`
Expected: PASS

### Task 3: Broader targeted verification

**Files:**
- Test: `source/isaaclab_rl/test/test_smp_runner_style_program.py`
- Test: `source/isaaclab_rl/test/test_smp_reward_logging.py`

- [ ] **Step 1: Run targeted regression tests**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: PASS
