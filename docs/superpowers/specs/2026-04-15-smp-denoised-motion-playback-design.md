# SMP Denoised Motion Playback Design

**Date:** 2026-04-15

## Goal

新增一个 Isaac Lab 脚本，输入训练用的 G1 SMP 数据集 `npz`（或 corpus 目录），截取其中第 5 到第 6 秒的 `frames` 片段，按 diffusion prior 训练时相同的前向加噪公式生成 `x_t`，再利用给定 checkpoint 的扩散模型预测噪声并重建去噪后的特征序列，最后在 Isaac Sim 中仅播放去噪片段，用于快速验证 prior 的恢复质量。

## User Requirements

- 输入是训练用的原始 `npz` 数据，当前样例位于 `/home/lucas/whole_body_tracking/artifacts/combine/g1_walk_corpus`
- 支持通过命令行切换不同的 diffusion checkpoint，而不是写死模型路径
- 只需要截取第 5 到第 6 秒的一秒片段进行验证
- Isaac 中只播放去噪片段，不需要并排对比原始或加噪片段

## Existing Code Context

当前仓库已经具备实现该工具所需的大部分基础设施：

- `scripts/imitation_learning/smp/train_motion_prior.py`
  - 说明 prior 训练入口，确认训练阶段使用 `rsl_rl.diffusion.SMPDiffusionTrainer`
- `rsl_rl/rsl_rl/diffusion/scheduler.py`
  - 提供 `DiffusionScheduler.q_sample()`，可直接复用训练时的前向加噪公式
- `rsl_rl/rsl_rl/diffusion/sampler.py`
  - 提供 `SMPDiffusionSampler`，但它更偏向从纯噪声开始完整反向采样
- `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
  - 提供 `_load_prior_model()`，可复用 checkpoint 的推理配置恢复逻辑
- `rsl_rl/rsl_rl/diffusion/gsi.py`
  - 提供 `SMPFeatureLayout`、rot6d 解码及 reset-state 相关还原逻辑，可复用其中的 joint rot6d 解码数学
- `source/isaaclab_tasks/.../mdp/smp_features.py`
  - 定义了当前 G1 SMP frame 的特征布局：
    - `base_lin_vel_b`
    - `base_ang_vel_b`
    - `joint_rot6d_rel`
    - `ee_pos_b`
- `scripts/demos/bipeds.py`
  - 提供了最小 G1 资产加载与仿真循环范式，适合作为播放脚本的基础

## Important Constraint

当前数据集中的 `frames` 不是 environment action，也不包含完整绝对根姿态轨迹。它只包含 heading-local 根速度、相对默认姿态的 joint rot6d，以及末端执行器相对位置。

这意味着：

- 可以稳定恢复：
  - 29 个关节角
  - 根线速度与角速度
- 不能严格恢复：
  - 原始绝对 root pose
  - 完整根朝向历史
  - 训练数据中的真实控制 action

因此，最终播放不是“原始动作精确复现”，而是“根据去噪后的特征做近似轨迹重建后的可视化验证”。这满足“验证 diffusion 恢复效果”的目标，但需要在脚本文档中明确说明。

## Design Choice

选择“窗口加噪 + 直接重建 `x0_hat` + 连续片段播放”的方案，而不是使用纯噪声完整采样。

原因：

1. 最贴近训练目标
   - 训练时模型学习的是从 `x_t` 预测噪声 `eps`，因此最直接的验证方式是给定真实 `x_0` 生成 `x_t`，再检查模型能否恢复 `x_0`
2. 更稳定
   - 完整反向采样会引入更高随机性和更明显的窗口间不连续问题，不利于做固定片段验证
3. 更适合用户当前需求
   - 用户希望针对已有数据做“训练效果验证播放”，而不是生成全新 motion

## Script Responsibilities

新增脚本暂定为：

- `scripts/imitation_learning/smp/play_denoised_motion_clip.py`

该脚本负责：

1. 加载输入数据集（单个 `npz` 或 corpus 目录）
2. 选择目标 clip（通过 `--source-name` 或直接传单文件）
3. 按秒数裁剪 5–6 秒片段
4. 加载 diffusion prior checkpoint
5. 按训练公式对每个 motion window 加噪
6. 用 prior 预测噪声并重建 `x0_hat`
7. 对重叠窗口进行 overlap-average 融合，恢复连续片段
8. 将去噪特征解码为 Isaac 可播放的近似机器人状态轨迹
9. 在 Isaac 中逐帧播放去噪片段
10. 打印基础误差统计，并可选保存调试结果

## CLI Design

脚本支持以下核心参数：

- `--dataset`
  - 必填；可以是单个 `dataset_xxx.npz` 或 corpus 目录
- `--checkpoint`
  - 必填；diffusion prior checkpoint 路径
- `--source-name`
  - 可选；当 `--dataset` 是目录时，用于选择某个 clip（如 `walk1`）
- `--clip-start-sec`
  - 可选；默认从 5.0 秒开始，避免总是取开头
- `--clip-duration-sec`
  - 可选；默认 1.0 秒，即播放第 5 到第 6 秒
- `--noise-step`
  - 可选；指定固定加噪 timestep，默认中间步（例如 15）
- `--device`
  - 可选；显式指定 `cpu` / `cuda:0`
- `--save-debug-npz`
  - 可选；保存 `original/noisy/denoised` 片段到调试文件
- `--play-source`
  - 可选；默认 `denoised`，可切换为 `clean`、`noisy`、`reference`
- `--lock-root`
  - 可选；固定 root，仅播放关节，用来判断扭曲是否来自 root 重建
- `--reference-motion-npz`
  - 可选；传入 `csv_to_npz.py` 生成的原始 state npz，配合 `--play-source reference` 做精确原始状态回放
- `--robot-source`
  - 可选；`auto` / `isaaclab` / `wbt`
  - `auto` 在 `reference` 模式下自动切到 `wbt`，其他模式默认 `isaaclab`
- `--render-only`
  - 可选；只写状态并渲染，不执行 `sim.step()`，用于和 `csv_to_npz.py` 的原始回放方式保持一致
- `--headless`
  - 继承 Isaac AppLauncher 标准参数

## Data Flow

### 1. Input Resolution

- 若 `--dataset` 指向单个 `npz`，直接读取该文件
- 若指向目录：
  - 若提供 `--source-name`，优先按 `source_name` 匹配
  - 否则默认读取目录中的第一个 `dataset_*.npz`

### 2. Clip Slicing

- 读取 `fps`，当前 corpus 为 50 FPS
- 将 `clip_start_sec` 和 `clip_duration_sec` 转为 frame index
- 从 `frames[T, F]` 中裁出 `[clip_start : clip_end]`
- 若片段过短，报错并提示可用长度

### 3. Window Construction

- 使用 checkpoint / 数据集声明的 `window_size`（当前为 10）
- 用 stride=1 在截取片段上构建滑动窗口 `[N, W, F]`
- 每个窗口作为一次验证样本

### 4. Forward Diffusion

对每个窗口执行：

- `eps = randn_like(x0)`
- `xt = scheduler.q_sample(x0, t, eps)`

其中：

- `t` 由 `--noise-step` 指定为固定时间步，便于可重复比较
- 后续如果需要再扩展随机 timestep 模式

### 5. Denoising Reconstruction

模型输出 `eps_hat = model(xt, t, style_id)` 后，用 DDPM 的 `x0` 重建公式：

- `x0_hat = (xt - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)`

说明：

- 这里优先做单步 `x0_hat` 重建，而不是完整反向链采样
- 如果 checkpoint 包含 style head，并且输入数据携带 `style_id`，则自动传入对应条件
- 若用户后续想测 unconditional，可在 CLI 中再扩展覆盖选项

### 6. Window Stitching

由于每帧会出现在多个窗口中，需要把多个 `x0_hat` 融合成一段连续片段：

- 对每个窗口对应的时间范围累加预测结果
- 对每个时间索引统计覆盖次数
- 最终逐帧除以覆盖次数，得到 overlap-average 后的 `denoised_frames[T_clip, F]`

这样可以显著减少窗口边界抖动。

### 7. Playback State Reconstruction

从每帧特征中恢复用于播放的近似状态：

- `joint_rot6d_rel -> joint angle offsets -> joint_pos`
- `base_lin_vel_b/base_ang_vel_b` 在 heading frame 下给出根速度
- 近似恢复策略：
  - 初始 root pose 取 G1 默认姿态
  - 初始 yaw = 0
  - 通过 `base_ang_vel_b[..., 1]` 的近似 heading yaw 分量进行逐帧积分
  - 通过当前 heading 将 `base_lin_vel_b` 旋回世界系，再对位置积分
  - 保持 root 高度为默认高度
  - 保持机器人 roll/pitch 为 0，仅使用积分得到的 yaw 构造根朝向
- `joint_vel` 默认置零，或由差分 joint_pos 近似求得

这是一个“视觉验证轨迹”，不是严格物理一致的状态重放。

## Isaac Playback Strategy

播放不通过 environment action，而是直接写机器人状态：

- 加载单个 G1 机器人
- 每个仿真步：
  - 写入 root pose / root velocity
  - 写入 joint position / joint velocity
  - 同步 `set_joint_position_target()`
  - `sim.step()`
  - `robot.update(sim_dt)`

该策略与仓库中的 `scripts/demos/bipeds.py`、`run_articulation.py` 保持一致。

## Metrics and Logging

脚本在终端打印以下统计：

- clip 来源、时长、fps、frame 范围
- checkpoint 路径
- noise timestep
- 整段片段的：
  - `noisy_vs_clean_mse`
  - `denoised_vs_clean_mse`
  - `joint_block_mse`
  - `velocity_block_mse`

如果开启 `--save-debug-npz`，则保存：

- `clean_frames`
- `noisy_frames`
- `denoised_frames`
- `fps`
- `noise_step`
- `source_name`

## Testing Strategy

实现采用 TDD，优先给纯 Python 部分补单测，不对 Isaac GUI 播放层做自动化 UI 测试。

新增测试将覆盖：

1. 数据集解析
   - 单文件与目录模式
   - `source_name` 选择逻辑
2. clip 裁剪
   - 秒到帧换算
   - 越界报错
3. window 构建与 overlap-average
   - 融合后长度与数值正确
4. `x0_hat` 重建公式
   - 使用假 scheduler/model 验证数学正确性
5. frame 解码
   - `joint_rot6d_rel` 正确恢复 joint positions
   - heading/yaw 积分输出维度与基本数值合理

## Non-Goals

本次不做：

- 原始 action 级别回放
- 原始绝对 motion 轨迹精确重建
- 多机器人并排对比显示
- 完整反向扩散生成全新 motion
- 在线环境 reward / RL runner 集成

## Implementation Notes

为了避免复制 runner 里过多私有逻辑，建议在新脚本内抽出一个小型 prior 加载辅助函数：

- 只依赖 checkpoint 中的 `model_cfg` 与 `style_cfg`
- 恢复 `MotionEpsilonTransformer`
- 返回 `model`、`num_diffusion_steps`、`style_id`

同时，把与播放无关的纯数学逻辑拆到一个可测试模块中，例如：

- `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py`
  - 数据集选择
  - clip 裁剪
  - window 构建/融合
  - prior 推理与 `x0_hat` 重建
  - 近似 playback state 重建

而 Isaac 脚本本身只负责：

- CLI 参数解析
- AppLauncher / SimulationContext 生命周期
- 调用 helper 产出状态序列并播放

## Success Criteria

满足以下条件即视为完成：

1. 用户可通过一条命令指定：
   - 数据集/语料目录
   - checkpoint
   - 起始秒数
   - 持续时长
   - noise timestep
2. 脚本能从 corpus 中成功截取片段并完成加噪/去噪
3. Isaac Sim 中能看到去噪后的 G1 片段播放
4. 终端能输出去噪前后相对 clean clip 的误差指标
5. 纯计算逻辑有单测覆盖并能在本地通过

## Run Command

示例命令：

```bash
./isaaclab.sh -p scripts/imitation_learning/smp/play_denoised_motion_clip.py \
  --dataset /home/lucas/whole_body_tracking/artifacts/combine/g1_walk_corpus \
  --source-name walk1 \
  --checkpoint /home/lucas/isaac-sim/IsaacLab/logs/smp_prior/g1/pretrain_407/model_latest.pt \
  --clip-start-sec 5.0 \
  --clip-duration-sec 1.0 \
  --noise-step 15
```

脚本默认使用所选数据集的 `fps` 作为仿真步长，只播放去噪后的 G1 29DOF 片段；默认截取 `[5.0s, 6.0s)`。添加 `--play-once` 可播放一遍后退出，添加 `--save-debug-npz /tmp/smp_debug.npz` 可保存 clean/noisy/denoised 特征用于离线检查。`reference` 模式默认走 `whole_body_tracking` 的 `G1_CYLINDER_CFG`，并强制 `render-only`，以尽量复现 `csv_to_npz.py` 当时的状态回放条件。

诊断命令：

```bash
# 播放 clean 特征经同一近似 decoder 重建后的结果；如果仍扭曲，说明不是 diffusion prior 问题
./isaaclab.sh -p scripts/imitation_learning/smp/play_denoised_motion_clip.py \
  --dataset /home/lucas/whole_body_tracking/artifacts/combine/g1_walk_corpus \
  --source-name walk1 \
  --checkpoint /home/lucas/isaac-sim/IsaacLab/logs/smp_prior/g1/pretrain_407/model_latest.pt \
  --clip-start-sec 5.0 \
  --clip-duration-sec 1.0 \
  --noise-step 0 \
  --play-source clean

# 固定 root 只播放关节；如果明显改善，说明 root/yaw 重建是主要问题
./isaaclab.sh -p scripts/imitation_learning/smp/play_denoised_motion_clip.py \
  --dataset /home/lucas/whole_body_tracking/artifacts/combine/g1_walk_corpus \
  --source-name walk1 \
  --checkpoint /home/lucas/isaac-sim/IsaacLab/logs/smp_prior/g1/pretrain_407/model_latest.pt \
  --clip-start-sec 5.0 \
  --clip-duration-sec 1.0 \
  --noise-step 0 \
  --play-source clean \
  --lock-root

# 播放 csv_to_npz.py 生成的原始 state npz；这是最接近真实 LaFAN 转换结果的基线
./isaaclab.sh -p scripts/imitation_learning/smp/play_denoised_motion_clip.py \
  --play-source reference \
  --reference-motion-npz /home/lucas/whole_body_tracking/artifacts/amp_walk1:v0/motion.npz \
  --clip-start-sec 5.0 \
  --clip-duration-sec 1.0

# 用 whole_body_tracking 的原始 G1 资产 + render-only 播放 clean；
# 如果这时不扭，问题更像是资产默认姿态/物理步进不一致
./isaaclab.sh -p scripts/imitation_learning/smp/play_denoised_motion_clip.py \
  --dataset /home/lucas/whole_body_tracking/artifacts/combine/g1_walk_corpus \
  --source-name walk1 \
  --checkpoint /home/lucas/isaac-sim/IsaacLab/logs/smp_prior/g1/pretrain_407/model_latest.pt \
  --clip-start-sec 5.0 \
  --clip-duration-sec 1.0 \
  --noise-step 0 \
  --play-source clean \
  --robot-source wbt \
  --render-only
```
