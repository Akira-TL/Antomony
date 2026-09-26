# Designs

## Objects

- [反馈时间范围比较](feedback-window.md) - 已冻结；同状态只比较16与64步一次。

- [基础扰动恢复参照](restoration-control.md) - 已冻结；区分改善机会不足与候选未找到改善。

- [无危险基础更新学习](basic-update-learning.md) - 已冻结；固定训练与留出种子，不接触危险机制。
- [行动概率约束候选诊断](trust-candidate-value.md) - 已冻结；最多64点，检验后续实际差异，不训练接受网络。
- [候选更新短期价值诊断](candidate-update-value.md) - 已冻结；采样72点，结论边界见描述分析，不作优势检验。
- [蚁群自主参数更新与未见环境适应性对照](ant-self-training-adaptation.md) - `draft`；当前主线，接受决策、对照能力与确认环境尚未齐备。
- [小型神经预测器的近期更新选择性回退](neural-readout-selective-rollback.md) — draft；固定特征、线性输出层、解析回退候选与后到反馈确认。生成参数和最终判据尚待固定，未执行。

## Relations

- [反馈时间范围](feedback-window.md) → [实际采样](../study/feedback-window/README.md)、[恢复诊断](../analysis/restoration-control/README.md)。后者作为课程及候选问题的开发依据。

- [恢复参照](restoration-control.md) → [实际采样](../study/restoration-control/README.md)、[本次分析](../analysis/restoration-control/README.md)、[前次基础训练结果](../analysis/basic-update-learning/README.md)。后者作为新诊断动机，不重用旧留出调参。

- [基础更新学习](basic-update-learning.md) → [实际采样](../study/basic-update-learning/README.md)、[本次分析](../analysis/basic-update-learning/README.md)、[既有候选诊断](../analysis/trust-candidate-value/README.md)。后者仅作为开发动机，不将其危险数据用于训练。
- [概率约束诊断](trust-candidate-value.md) → [实际采样](../study/trust-candidate-value/README.md)、[描述分析](../analysis/trust-candidate-value/README.md)。
- [候选更新诊断](candidate-update-value.md) → [适应性总设计](ant-self-training-adaptation.md)、[实际采样](../study/candidate-update-value/README.md)、[描述分析](../analysis/candidate-update-value/README.md)。
- [蚁群对照设计](ant-self-training-adaptation.md) → [当前研究状态](../RESEARCH.md)。问题直接驱动，尚无正式实施或分析对象。
- [选择性回退设计](neural-readout-selective-rollback.md) → [研究问题与原模块边界](../docs/competition/v2v-extraction-scope.md)、[当前研究状态](../RESEARCH.md)。当前没有关联的假设集合、研究实施或分析结果。
