# SMP Target-vs-Uncond Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable `target_vs_uncond` SMP reward mode that changes only the raw SMP reward calculation while preserving downstream reward scaling and combination.

**Architecture:** Extend `SMPReward` with a `reward_mode` switch. Keep the existing EMA normalization method intact, but allow differential reward computation by normalizing target and unconditioned MSE streams with the same method under separate running states, then computing reward from their difference. Update the SMP runner to pass both target-style and unconditioned predictions only when needed, leaving reward coefficients and reward merging untouched.

**Tech Stack:** Python, PyTorch, pytest, config dataclasses

---

### Task 1: Add failing tests for reward modes

**Files:**
- Modify: `source/isaaclab_rl/test/test_smp_reward_logging.py`

- [ ] **Step 1: Write the failing tests**

Add tests that verify:
- `absolute` mode preserves the current behavior
- `target_vs_uncond` mode gives a higher reward when `eps_cond` is closer to `eps` than `eps_uncond`

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest source/isaaclab_rl/test/test_smp_reward_logging.py -k 'smp_reward_uses_fixed_timestep_ensemble or target_vs_uncond' -q`
Expected: FAIL because `SMPReward` does not yet accept or implement `reward_mode`.

### Task 2: Add configurable reward mode

**Files:**
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py`
- Modify: `rsl_rl/rsl_rl/diffusion/smp_reward.py`

- [ ] **Step 1: Add config field**

Add a `reward_mode` config entry with default `absolute`.

- [ ] **Step 2: Implement reward mode logic**

Update `SMPReward` to:
- validate `reward_mode`
- preserve the existing normalization method
- support `absolute`
- support `target_vs_uncond` using target and unconditioned predictions

- [ ] **Step 3: Run reward tests**

Run: `pytest source/isaaclab_rl/test/test_smp_reward_logging.py -k 'smp_reward_uses_fixed_timestep_ensemble or target_vs_uncond' -q`
Expected: PASS

### Task 3: Wire runner inputs for differential reward

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- Modify: `source/isaaclab_rl/test/test_smp_runner_style_program.py`

- [ ] **Step 1: Add failing runner test**

Add a focused runner test to verify the differential reward path can consume target and unconditioned predictions without affecting downstream combination.

- [ ] **Step 2: Implement runner wiring**

Update the runner so that:
- `absolute` mode keeps the current path
- `target_vs_uncond` passes both target-style and unconditioned eps to `SMPReward`
- downstream coefficient multiplication and reward merge remain unchanged

- [ ] **Step 3: Run runner test**

Run: `pytest source/isaaclab_rl/test/test_smp_runner_style_program.py -k target_vs_uncond -q`
Expected: PASS

### Task 4: Targeted verification

**Files:**
- Test: `source/isaaclab_rl/test/test_smp_reward_logging.py`
- Test: `source/isaaclab_rl/test/test_smp_runner_style_program.py`

- [ ] **Step 1: Run targeted regression tests**

Run: `pytest source/isaaclab_rl/test/test_smp_reward_logging.py source/isaaclab_rl/test/test_smp_runner_style_program.py -q`
Expected: PASS
