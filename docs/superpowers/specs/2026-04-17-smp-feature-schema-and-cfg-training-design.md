# SMP Feature Schema And CFG Training Design

## Goal

在保持旧版 `192D` SMP 数据、扩散 checkpoint 和 RL 配置继续可用的前提下，扩展整条 SMP 链路，使其支持：

- 使用多风格数据训练**单一** `style-conditioned` diffusion 模型；
- 通过 `style_drop_prob` 学习 classifier-free guidance 所需的 `uncond` 与 `cond` 预测；
- 在原有 `192D` 特征后追加 `base_lin_vel_w(3)` 和 `base_ang_vel_w(3)`，形成新的 `198D` 特征；
- 确保新导出的 paired NPZ 仍然能够稳定播放；
- 让 CSV 转 NPZ、paired dataset、diffusion 训练、RL 观测窗口、prior runner 与诊断工具都通过统一接口选择 `192D` 或 `198D`。

## Training Direction

本次设计采用与 SMP 论文一致的推荐路线：**一个共享参数的 style-conditioned diffusion 模型**，不是两个独立模型。

训练阶段：

- 数据集提供真实 `style_id`；
- trainer 对一部分样本执行 style dropout，把其标签替换为 `NULL_STYLE_ID`；
- 同一个模型在同一训练流程中同时学习：
  - `f(x_i, c)`：带真实风格标签的条件噪声预测；
  - `f(x_i, ∅)`：null-style 的无条件噪声预测。

推理阶段：

- prior runner 使用同一个冻结模型分别计算 `f(x_i, ∅)` 与 `f(x_i, c)`；
- 然后按 CFG 公式组合成 style-specific prior。

这一路线保留现有实现的大部分骨架，只需要把 `style_drop_prob` 从“可选统计项”提升为“推荐默认训练路径”。

## Feature Schemas

首版定义两种显式 schema：

- `legacy_192`
- `extended_198`

### `legacy_192`

保持现状：

- `base_lin_vel_b`: `3`
- `base_ang_vel_b`: `3`
- `joint_rot6d_rel`: `29 * 6 = 174`
- `ee_pos_b`: `4 * 3 = 12`

总计 `192`。

### `extended_198`

保持旧 `192D` 前缀**完全不变**，然后在末尾追加：

- `base_lin_vel_w`: `3`
- `base_ang_vel_w`: `3`

总计 `198`。

追加顺序固定如下：

1. 旧 `192D`
2. `base_lin_vel_w`
3. `base_ang_vel_w`

这样可以最大程度兼容旧逻辑：

- 旧模型和旧工具如果只理解前 `192` 维，不会因为原始块顺序变化而失语义；
- 新工具可以通过 metadata 明确知道最后 `6` 维是什么。

## Dataset Metadata

paired NPZ 与普通训练 NPZ 都应显式记录特征布局，而不是仅凭 `feature_dim` 猜测。

### Required Metadata

在现有字段基础上新增：

- `feature_schema`

并继续维护：

- `feature_dim`
- `feature_block_names`
- `feature_block_offsets`

对于 `extended_198`，block 定义应包含：

- `base_lin_vel_b`
- `base_ang_vel_b`
- `joint_rot6d_rel`
- `ee_pos_b`
- `base_lin_vel_w`
- `base_ang_vel_w`

对于 `legacy_192`，保持现有四个 block。

通过这套 metadata：

- 回放脚本可以知道自己拿到的是 `192` 还是 `198`；
- style composition / body mask 等依赖 block offsets 的逻辑仍能工作；
- 诊断与报错可以更明确。

## CSV To NPZ Conversion

修改 `scripts/imitation_learning/smp/csv_to_smp_paired_dataset.py`，增加显式接口：

- `--feature-schema {legacy_192,extended_198}`

### Conversion Rules

- `legacy_192`：
  - 导出当前已有的 `192D` `frames`
- `extended_198`：
  - 保留旧 `192D` 排列；
  - 在末尾追加 `base_lin_vel_w` 与 `base_ang_vel_w`

脚本继续保存 paired dataset 当前已经携带的 raw state：

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

这保证新导出的 NPZ 仍可用于 raw replay。

## Playback Compatibility

`scripts/imitation_learning/smp/replay_smp_paired_dataset.py` 必须继续支持播放新导出的 paired NPZ。

本设计中，回放稳定性的保证来自两个层面：

- **主回放路径继续基于 raw state**，而不是依赖 `frames` 反解；
- paired dataset 仍保存完整 replay 所需 raw state 字段。

因此：

- `legacy_192` paired NPZ 可播放；
- `extended_198` paired NPZ 也可播放；
- 播放逻辑不应因为 `frames.shape[-1] == 198` 而失败。

若脚本内部对 `frames` 做辅助打印、校验或未来扩展，应统一改为读取：

- `feature_schema`
- `feature_dim`
- `feature_block_offsets`

而不是假设固定 `192`。

## Shared Feature Packing

本次改动的核心是把 SMP 特征打包规范抽象成 schema-aware 的统一逻辑。

### Files

- `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_paired_dataset.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`

### Required Changes

- 将 block offset 计算逻辑改成支持 schema；
- 将特征组件构造逻辑扩展到同时产出：
  - heading-local `base_lin_vel_b`
  - heading-local `base_ang_vel_b`
  - world-frame `base_lin_vel_w`
  - world-frame `base_ang_vel_w`
- 将 `pack_smp_frame_features(...)` 改为根据 schema 决定输出 `192` 或 `198`；
- 所有写死 `192` 的维度断言改为：
  - 读取 schema 后得到期望 feature dim；
  - 或从 metadata / 配置显式传入。

`extended_198` 的 world velocity 不改变原有 `192D` 各块的坐标系与语义，仅作为追加信息供 prior 与 policy 使用。

## RL Motion Window

环境侧 `smp_motion_window` 需要与 dataset / checkpoint 使用相同 schema。

### Files

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`

### Required Behavior

- 增加 `feature_schema` 配置接口；
- `legacy_192` 时维持现有观测；
- `extended_198` 时，在当前 `192D` 观测末尾追加：
  - `asset.data.root_lin_vel_w`
  - `asset.data.root_ang_vel_w`

环境配置不应只暴露一个固定的 `g1_smp_feature_dim = 192`，而应提供按 schema 计算的：

- feature dim
- feature block offsets

这样 RL 环境生成的窗口、paired NPZ 的 `frames`、prior checkpoint 的输入维度才会严格对齐。

## Diffusion Training

### Files

- `rsl_rl/rsl_rl/motion/smp_dataset.py`
- `rsl_rl/rsl_rl/diffusion/trainer.py`
- `scripts/imitation_learning/smp/train_motion_prior.py`

### Dataset Loader

`SMPMotionWindowDataset` 继续直接加载 `frames`，不把 feature dim 写死为 `192`。同时增加对 `feature_schema` 的读取与暴露，便于日志和调试。

### Trainer

保留当前单模型训练结构：

- `feature_dim` 由 dataset 推断；
- `style_id` 来自 batch；
- `maybe_drop_style(...)` 决定哪些样本走 `NULL_STYLE_ID`；
- 同一个模型统一预测噪声。

不引入第二个模型，也不引入每 batch 双前向双 loss 的新训练范式。

### CLI Semantics

`train_motion_prior.py` 保留 `--style-drop-prob`，但要把它从“仅供实验的可选参数”提升为“正常使用 CFG 的关键训练开关”。

首版建议：

- 允许用户显式设置；
- 若检测到 dataset 存在 style label 且 `style_drop_prob == 0.0`，给出清晰警告；
- 文档与帮助文本中说明：
  - `0.0` 基本只训练条件分支；
  - 非零值才真正训练 null-style / uncond 分支。

## Prior Runner And Diagnostics

### Files

- `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_diagnostics.py`
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_denoising_diagnostic.py`
- 其他依赖 prior `feature_dim` 的 helper

### Required Behavior

- 统一以 checkpoint / config 中的 `feature_dim` 与 `feature_schema` 为准；
- `_restore_smp_window(...)`、diagnostic loader、window collector 等都不得假设 `192`；
- 发现以下不一致时应尽早报错：
  - env 观测窗口维度与 prior checkpoint 不一致；
  - dataset `frames` 维度与 checkpoint 不一致；
  - style composition 的 block offsets 与 feature dim 不一致。

这样可以避免“加载成功但数值语义错位”的隐蔽错误。

## Compatibility Strategy

### Old Assets Continue To Work

- 旧 `192D` dataset：继续可训练、可回放、可做 prior 诊断；
- 旧 `192D` checkpoint：继续可加载并用于 `192D` RL 窗口；
- 旧 RL 配置：默认保持 `legacy_192`。

### New Assets Require Explicit Opt-In

只有在显式设置 `feature_schema=extended_198` 时，以下链路才切到 `198D`：

- CSV 导出
- paired dataset
- RL `smp_motion_window`
- diffusion prior 训练
- prior checkpoint
- SMP reward / diagnostic

这可防止现有实验被隐式升级到新特征维度。

## Error Handling

首版必须优先加强这些错误提示：

- dataset `feature_schema` 缺失但 `feature_dim` 既不是已知 `192` 也不是已知 `198`
- env feature dim 与 prior checkpoint feature dim 不一致
- `feature_block_offsets` 超出 `feature_dim`
- `extended_198` 数据缺失 `base_lin_vel_w` 或 `base_ang_vel_w` block metadata
- replay 输入 paired NPZ 缺失 raw state 字段

错误信息应直接指出：

- 当前读到的 schema / dim
- 期望的 schema / dim
- 建议检查的配置项或文件

## Testing

### Unit Tests

扩展现有测试，优先覆盖：

- `source/isaaclab_rl/test/test_smp_feature_utils.py`
  - `legacy_192` 输出维度
  - `extended_198` 输出维度
  - 追加顺序固定为末尾 `6` 维
- `source/isaaclab_rl/test/test_smp_dataset.py`
  - 加载 `192D` 与 `198D` `frames`
  - 读取 `feature_schema`
- `source/isaaclab_rl/test/test_smp_paired_dataset.py`
  - paired dataset 保存/加载 `feature_schema`
  - `extended_198` block metadata 正确
- `source/isaaclab_rl/test/test_smp_runner_style_program.py`
  - runner 恢复不同 feature dim 的窗口
  - 不匹配时清晰报错

### Script-Level Verification

至少验证：

- `csv_to_smp_paired_dataset.py --help`
- `replay_smp_paired_dataset.py --help`
- 新导出的 `extended_198` paired NPZ 能被 replay loader 成功读取

若环境允许，再做一次最小 smoke test：

- 导出一个 `extended_198` paired NPZ；
- 用 replay 脚本单机器人播放；
- 确认能跑通到渲染/物理步进阶段。

## Out Of Scope

首版不做：

- 为旧 `192D` checkpoint 自动升级到 `198D`
- 自动把 `198D` dataset 降级裁剪回 `192D`
- 引入第二个 unconditional diffusion 模型
- 重构 SMP reward 数学形式
- 修改 CFG 推理公式本身
- 改变旧 `192D` 特征块的顺序或坐标系定义
