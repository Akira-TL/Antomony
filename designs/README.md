# Designs

## Objects

- [生存反馈有限对照](survival-feedback.md) - 固定两个新种子16世界，先验证固定接受；生存优先、搬运次要。

- [历史奖励基线对照](historical-baseline.md) - 固定两个新种子20世界，只比较0与0.1，不搜索更多速率。

- [伤害后的单次候选诊断](injury-candidate.md) - 已冻结；最多32对，分支支持复活但不继续更新。

- [空载返巢奖励配对诊断](return-reward-ablation.md) - 已冻结并登记；固定两个新种子、20世界，只改变返巢奖励。

- [复杂来源连续更新对照](continuous-adaptation.md) - 已冻结并登记；固定80世界，主要比较同初始化的三种更新方式。

- [近似规模MLP基础资格](matched-foundation.md) - 已冻结；固定训练预算及48世界评价。

- [保留基础能力后的接受学习](memory-update-learning.md) - 已冻结；无危险课程，8训练与4留出种子，不使用未校准价值输出。

- [无线索探索课程](exploration-course.md) - 已冻结；32回合、固定最终模型，不挑中间点。

- [三组基础能力检查](foundation-qualification.md) - 已冻结；修正规则后固定48世界，不训练。

- [反馈时间范围比较](feedback-window.md) - 已冻结；同状态只比较16与64步一次。

- [基础扰动恢复参照](restoration-control.md) - 已冻结；区分改善机会不足与候选未找到改善。

- [无危险基础更新学习](basic-update-learning.md) - 已冻结；固定训练与留出种子，不接触危险机制。
- [行动概率约束候选诊断](trust-candidate-value.md) - 已冻结；最多64点，检验后续实际差异，不训练接受网络。
- [候选更新短期价值诊断](candidate-update-value.md) - 已冻结；采样72点，结论边界见描述分析，不作优势检验。
- [蚁群自主参数更新与未见环境适应性对照](ant-self-training-adaptation.md) - `draft`；当前主线，接受决策、对照能力与确认环境尚未齐备。
- [小型神经预测器的近期更新选择性回退](neural-readout-selective-rollback.md) — draft；固定特征、线性输出层、解析回退候选与后到反馈确认。生成参数和最终判据尚待固定，未执行。

## Relations

- [历史基线设计](historical-baseline.md) → [实施](../study/historical-baseline/README.md)、[分析](../analysis/historical-baseline/README.md)。

- [伤害候选诊断](injury-candidate.md) → [实际实施](../study/injury-candidate/README.md)、[配对分析](../analysis/injury-candidate/README.md)。

- [返巢奖励诊断](return-reward-ablation.md) → [实际实施](../study/return-reward-ablation/README.md)、[配对分析](../analysis/return-reward-ablation/README.md)。

- [连续对照](continuous-adaptation.md) → [基础资格](../analysis/matched-foundation/README.md)、[固定接受模型](../data/memory-acceptance-models/README.md)、[实际实施](../study/continuous-adaptation/README.md)、[连续数据](../data/continuous-adaptation/README.md)、[连续分析](../analysis/continuous-adaptation/README.md)。

- [近似规模基础资格](matched-foundation.md) → [实际实施](../study/matched-foundation/README.md)、[模型与轨迹](../data/matched-foundation/README.md)、[资格分析](../analysis/matched-foundation/README.md)。

- [新基础接受课程](memory-update-learning.md) → [实际采样](../study/memory-update-learning/README.md)、[原始数据](../data/memory-update-learning/README.md)、[接受学习结果](../analysis/memory-update-learning/README.md)。

- [探索课程](exploration-course.md) → [实际实施](../study/exploration-course/README.md)、[配对评价](../analysis/exploration-course/README.md)。

- [三组基础检查](foundation-qualification.md) → [实际采样](../study/foundation-qualification/README.md)、[描述分析](../analysis/foundation-qualification/README.md)。

- [反馈时间范围](feedback-window.md) → [实际采样](../study/feedback-window/README.md)、[本次分析](../analysis/feedback-window/README.md)、[恢复诊断](../analysis/restoration-control/README.md)。后者作为课程及候选问题的开发依据。

- [恢复参照](restoration-control.md) → [实际采样](../study/restoration-control/README.md)、[本次分析](../analysis/restoration-control/README.md)、[前次基础训练结果](../analysis/basic-update-learning/README.md)。后者作为新诊断动机，不重用旧留出调参。

- [基础更新学习](basic-update-learning.md) → [实际采样](../study/basic-update-learning/README.md)、[本次分析](../analysis/basic-update-learning/README.md)、[既有候选诊断](../analysis/trust-candidate-value/README.md)。后者仅作为开发动机，不将其危险数据用于训练。
- [概率约束诊断](trust-candidate-value.md) → [实际采样](../study/trust-candidate-value/README.md)、[描述分析](../analysis/trust-candidate-value/README.md)。
- [候选更新诊断](candidate-update-value.md) → [适应性总设计](ant-self-training-adaptation.md)、[实际采样](../study/candidate-update-value/README.md)、[描述分析](../analysis/candidate-update-value/README.md)。
- [蚁群对照设计](ant-self-training-adaptation.md) → [当前研究状态](../RESEARCH.md)。问题直接驱动，尚无正式实施或分析对象。
- [选择性回退设计](neural-readout-selective-rollback.md) → [研究问题与原模块边界](../docs/competition/v2v-extraction-scope.md)、[当前研究状态](../RESEARCH.md)。当前没有关联的假设集合、研究实施或分析结果。
