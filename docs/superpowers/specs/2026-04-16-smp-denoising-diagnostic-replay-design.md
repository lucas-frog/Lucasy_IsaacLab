# SMP Denoising Diagnostic Replay Design

## Goal

提供一个独立的 Isaac Sim 可视化诊断脚本，用已经处理好的 paired NPZ 中的 `frames(192D)` 做扩散前向加噪与后向去噪，并像现有稳定 raw replay 一样直接在 Isaac 里单机器人回放 `raw / clean_decoded / noisy_decoded / denoised_decoded` 四种来源中的任意一种。

## Why Separate Tool

这个工具不复用 `play_denoised_motion_clip.py` 作为主入口。现有脚本已经被怀疑存在实现问题，而且它混合了 denoising、decoded playback、reference playback 等职责，不适合作为当前“验证扩散链路是否正常”的基线。

新的诊断脚本应当：

- 复用已经修好的 raw replay 写入路径，保证 `raw` 模式稳定；
- 把 `clean/noisy/denoised` 都明确视为“从 192D 解码得到的诊断预览”；
- 把扩散数值流程和 Isaac 回放逻辑隔离，便于后续分别检查。

## Input And Output

### Inputs

- `paired npz`
  - 包含 `frames`
  - 包含 raw replay 状态字段：`joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`
- 扩散 checkpoint
  - 当前目标路径：`/home/lucas/isaac-sim/IsaacLab/logs/smp_prior/g1/pretrain_407/model_latest.pt`

### Outputs

- Isaac Sim 单机器人回放
- 终端数值指标
- 可选 debug NPZ，保存 `clean/noisy/denoised` 三份 192D clip 结果与 metadata

## Playback Modes

脚本支持以下 `--play-source`：

- `raw`
  - 使用 paired NPZ 中保存的 raw 状态直接回放
  - 这是视觉基准真值
- `clean_decoded`
  - 从 clip 的 clean `frames(192D)` 解码后回放
- `noisy_decoded`
  - 从前向加噪后的 `frames(192D)` 解码后回放
- `denoised_decoded`
  - 从模型去噪得到的 `frames(192D)` 解码后回放

其中 decoded 路径全部视为诊断用途，不承诺和 raw 真值完全一致。

## Default Flow

脚本默认工作流如下：

1. 从 paired NPZ 加载 `frames` 与 raw 状态。
2. 按 `--clip-start-sec` / `--clip-duration-sec` 切出连续 clip。
3. 按 checkpoint 的 `window_size` 和 CLI `--stride` 构造窗口。
4. 通过 scheduler 的 `q_sample` 对窗口做前向加噪，噪声时间步由 `--noise-step` 指定。
5. 用 checkpoint 模型预测 `eps_hat`。
6. 用 DDPM 闭式公式从 `(x_t, eps_hat)` 重建 `x0_hat`。
7. 把窗口 stitch 回连续 clip。
8. 得到三份连续 192D clip：
   - clean
   - noisy
   - denoised
9. 把三者分别解码为可播放状态。
10. 按 `--play-source` 选一条在 Isaac 中单机器人回放。

## Decode Strategy

decoded playback 路径沿用已有 SMP 诊断解码思路，但明确限制边界：

- 根位置：通过 heading-local `base_lin_vel_b` 积分恢复平面位移
- 根朝向：通过 heading-local `base_ang_vel_b` 的 yaw 分量积分恢复 heading
- 关节角：通过 `joint_rot6d_rel` 解回相对默认姿态的关节角，再叠加默认关节位姿
- 末端位置 `ee_pos_b`：仅作为特征参与误差诊断，不做 IK 强制约束

因此 decoded 回放的职责是“看模型恢复的 192D 在运动学上是否大致合理”，不是复原严格真值。

## Raw Replay Baseline

`raw` 模式必须直接复用当前稳定的 paired raw replay 路径：

- 若 `joint_pos` 是完整机器人关节向量，则直接整向量写回
- 若将来存在只保存子集的 paired 数据，再按 `joint_names` 做映射写回

这保证 visual baseline 不会再被“joint order 错配”污染。

## Metrics

脚本至少打印以下诊断指标：

- `clean_vs_noisy_mse`
- `clean_vs_denoised_mse`
- `joint_block_mse`
- `velocity_block_mse`

这些指标与视觉回放一起使用：

- `raw` 正常、`denoised_decoded` 明显异常 → 优先怀疑解码或模型质量
- `clean_decoded` 已异常 → 优先怀疑 192D 解码逻辑
- `denoised_decoded` 比 `noisy_decoded` 更差 → 优先检查扩散推理链路

## CLI

首版 CLI 保持最小集合：

- `--dataset`
- `--checkpoint`
- `--clip-start-sec`
- `--clip-duration-sec`
- `--noise-step`
- `--play-source {raw,clean_decoded,noisy_decoded,denoised_decoded}`
- `--stride`
- `--style-id` 可选
- `--play-once`
- `--step-physics` 可选
- `--save-debug-npz` 可选

## Files

### New Script

- `scripts/imitation_learning/smp/replay_smp_denoising_diagnostic.py`

职责：

- 加载 paired NPZ 与 checkpoint
- 生成 clean/noisy/denoised clip
- 选择 raw 或 decoded 状态源
- 在 Isaac Sim 中单机器人回放

### Reused Helpers

- `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_paired_dataset.py`
  - raw clip 加载
  - paired dataset metadata 读取
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoised_playback.py`
  - clip slicing
  - motion windows
  - denoise / stitch
  - decoded playback state reconstruction

如复用过程中发现 helper 边界不清，可把纯数值部分下沉为更小的共享函数，但不修改 raw replay 主路径语义。

## Testing

先做聚焦单元测试，再做 CLI 帮助验证：

- 构造 toy `frames` + toy checkpoint mock，验证：
  - clip slicing
  - window denoise/stitch
  - `play-source` 选择逻辑
  - `raw` 模式优先走 paired raw replay
- CLI `--help`
- 如环境允许，再手动在 Isaac 中跑一段 clip 做 smoke test

## Out Of Scope

首版不做：

- 多机器人同屏比较
- 浏览器内可视化
- 自动生成视频
- IK 强约束末端修正
- 直接支持“只有纯 192D dataset，没有 raw 状态”的输入
