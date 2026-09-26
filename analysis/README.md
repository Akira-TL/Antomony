# Analyses

## Objects

- [生存反馈配对](survival-feedback/README.md) - planned，先审计后按冻结判据计算。

- [历史基线配对后果](historical-baseline/README.md) - 未通过采用条件；出现零交付但总奖励提高，需重新明确目标取舍。

- [伤害候选与接受判断](injury-candidate/README.md) - 已完成；两世界未一致出现有益候选被拒绝，候选质量及接受判断均有不足。

- [返巢奖励配对后果](return-reward-ablation/README.md) - 已完成；取消奖励未通过采用条件，保留当前默认值。

- [连续复杂来源对照](continuous-adaptation/README.md) - completed；更新优于跳过，但学习接受不如固定接受，未通过采用条件。

- [近似规模模型基础资格](matched-foundation/README.md) - 已完成；三组通过最低要求，探索与返回仍不同。

- [新基础接受学习](memory-update-learning/README.md) - 已完成；仅1/4种子优于两参照，没有所选焦点交付改善，未通过。

- [无线索课程最终评价](exploration-course/README.md) - 已完成；返回退步，未采用最终模型，不续训。

- [三组基础能力](foundation-qualification/README.md) - 已完成；三组搬运均完成，循环无线索探索未达开发门槛。

- [反馈时间范围比较](feedback-window/README.md) - 已完成；64步没有通过开发条件，保留16步，不追加窗口搜索。

- [恢复参照后果描述](restoration-control/README.md) - 已完成；同向扰动有可修正偏差，现有候选很少取得相应改善。

- [无危险基础更新学习](basic-update-learning/README.md) - 已完成，两个留出种子均不如全部跳过；保留全部模型及负结果。

- [行动概率约束候选后果描述](trust-candidate-value/README.md) - 已完成探索描述，58对中12对奖励提高、4对降低；接受决策仍未训练。
- [候选更新短期价值描述](candidate-update-value/README.md) - 已完成探索描述，保留数据库登记晚于结果的流程偏离。

## Relations

- [生存反馈](survival-feedback/README.md) → [设计](../designs/survival-feedback.md)、[数据](../data/survival-feedback/README.md)。

- [历史基线分析](historical-baseline/README.md) → [设计](../designs/historical-baseline.md)、[数据](../data/historical-baseline/README.md)。

- [伤害候选分析](injury-candidate/README.md) → [设计](../designs/injury-candidate.md)、[数据](../data/injury-candidate/README.md)。

- [返巢奖励配对后果](return-reward-ablation/README.md) → [冻结设计](../designs/return-reward-ablation.md)、[输入数据](../data/return-reward-ablation/README.md)。

- [连续对照分析](continuous-adaptation/README.md) → [冻结设计](../designs/continuous-adaptation.md)、[连续数据](../data/continuous-adaptation/README.md)。

- [基础接受分析](memory-update-learning/README.md) → [供连续部署的固定模型](../data/memory-acceptance-models/README.md)。

- [近似规模基础分析](matched-foundation/README.md) → [冻结设计](../designs/matched-foundation.md)、[原始数据](../data/matched-foundation/README.md)。

- [新基础接受学习](memory-update-learning/README.md) → [冻结设计](../designs/memory-update-learning.md)、[配对数据](../data/memory-update-learning/README.md)。

- [探索课程评价](exploration-course/README.md) → [冻结设计](../designs/exploration-course.md)、[原始数据](../data/exploration-course/README.md)。

- [基础能力分析](foundation-qualification/README.md) → [冻结设计](../designs/foundation-qualification.md)、[原始数据](../data/foundation-qualification/README.md)。

- [反馈范围分析](feedback-window/README.md) → [冻结设计](../designs/feedback-window.md)、[配对数据](../data/feedback-window/README.md)。

- [恢复参照描述](restoration-control/README.md) → [冻结设计](../designs/restoration-control.md)、[原始数据](../data/restoration-control/README.md)。

- [基础更新学习](basic-update-learning/README.md) → [冻结设计](../designs/basic-update-learning.md)、[配对数据](../data/basic-update-learning/README.md)。

- [概率约束后果描述](trust-candidate-value/README.md) → [冻结设计](../designs/trust-candidate-value.md)、[配对数据](../data/trust-candidate-value/README.md)。
- [本分析](candidate-update-value/README.md) → [冻结诊断设计](../designs/candidate-update-value.md)、[配对数据](../data/candidate-update-value/README.md)。
