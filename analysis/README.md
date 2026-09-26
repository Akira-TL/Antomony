# Analyses

## Objects

- [新基础接受学习](memory-update-learning/README.md) - 待执行；固定训练200步后评价四个留出种子。

- [无线索课程最终评价](exploration-course/README.md) - 已完成；返回退步，未采用最终模型，不续训。

- [三组基础能力](foundation-qualification/README.md) - 已完成；三组搬运均完成，循环无线索探索未达开发门槛。

- [反馈时间范围比较](feedback-window/README.md) - 已完成；64步没有通过开发条件，保留16步，不追加窗口搜索。

- [恢复参照后果描述](restoration-control/README.md) - 已完成；同向扰动有可修正偏差，现有候选很少取得相应改善。

- [无危险基础更新学习](basic-update-learning/README.md) - 已完成，两个留出种子均不如全部跳过；保留全部模型及负结果。

- [行动概率约束候选后果描述](trust-candidate-value/README.md) - 已完成探索描述，58对中12对奖励提高、4对降低；接受决策仍未训练。
- [候选更新短期价值描述](candidate-update-value/README.md) - 已完成探索描述，保留数据库登记晚于结果的流程偏离。

## Relations

- [新基础接受学习](memory-update-learning/README.md) → [冻结设计](../designs/memory-update-learning.md)、[配对数据](../data/memory-update-learning/README.md)。

- [探索课程评价](exploration-course/README.md) → [冻结设计](../designs/exploration-course.md)、[原始数据](../data/exploration-course/README.md)。

- [基础能力分析](foundation-qualification/README.md) → [冻结设计](../designs/foundation-qualification.md)、[原始数据](../data/foundation-qualification/README.md)。

- [反馈范围分析](feedback-window/README.md) → [冻结设计](../designs/feedback-window.md)、[配对数据](../data/feedback-window/README.md)。

- [恢复参照描述](restoration-control/README.md) → [冻结设计](../designs/restoration-control.md)、[原始数据](../data/restoration-control/README.md)。

- [基础更新学习](basic-update-learning/README.md) → [冻结设计](../designs/basic-update-learning.md)、[配对数据](../data/basic-update-learning/README.md)。

- [概率约束后果描述](trust-candidate-value/README.md) → [冻结设计](../designs/trust-candidate-value.md)、[配对数据](../data/trust-candidate-value/README.md)。
- [本分析](candidate-update-value/README.md) → [冻结诊断设计](../designs/candidate-update-value.md)、[配对数据](../data/candidate-update-value/README.md)。
