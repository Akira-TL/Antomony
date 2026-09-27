# Analyses

## Objects

- [同情境独立试行回报基线](plastic-paired-baseline/README.md) - 分析未完成；首次执行在配对情境审计处停止，未生成效果摘要。

- [方向归一化方式与训练前后配对评价](plastic-projection/README.md) - 已完成；两架构均一初始化全部跳过、另一全部接受，旧课程联合资格及新增两类增益均未通过，停止本配置。

- [可塑方向基础课程配对评价](feedback-plasticity-course/README.md) - 已完成；两初始化均全部接受，学得组与全部接受及等次数组奖励差为0，0/2通过；不延长、不接入8775。

- [基础候选交付与奖励覆盖](candidate-label-coverage/README.md) - 已完成；301候选无交付正例，固定集合二元选择上界0；已浏览标签后的探索，不拟合或修改旧判据。

- [首步方向32步后果](first-action-outcomes/README.md) - 已完成；8点每点16方向焦点指标相同，单次干预无判别力，停止扩样。

- [四步写入约束与动作响应](update-constraints/README.md) - completed；152/154写入实际动作总变差至少0.01，无角度饱和，不支持更新普遍不起作用的解释。

- [后到反馈动作归因](credit-history/README.md) - completed；54共同状态中新配置无有益候选，未通过，停止本路线。

- [四步候选动作输入学习](action-update-learning/README.md) - completed；训练及留出均无正收益候选，两输入0/2通过，停止本课程。

- [四步候选动作影响](candidate-steering/README.md) - completed，重建65536次动作；152次含受伤写入中70次即时朝向投影恶化，23次与目标方向变化相反。

- [四步与十六步窗口](window-cadence/README.md) - completed，四步比十六步少失败21/23次，但相对不更新多2/少50次，未通过两参照门槛。

- [负反馈提前判断](feedback-trigger/README.md) - completed；写入明显增加但相对固定窗口失败+1/+3，停止采用。

- [受伤与写入时序](survival-timing/README.md) - completed；26次危险死亡中21次此前已有受伤后写入，延迟不是唯一解释。

- [生存反馈配对](survival-feedback/README.md) - completed；相对不更新少失败6/58次，相对旧反馈多7/少21次，未通过采用。

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

- [同情境独立试行回报基线](plastic-paired-baseline/README.md) → [冻结设计](../designs/plastic-paired-baseline.md)、[固定数据](../data/plastic-paired-baseline/README.md)。

- [方向归一化配对评价](plastic-projection/README.md) → [冻结设计](../designs/plastic-projection.md)、[固定原始数据](../data/plastic-projection/README.md)、[真实实施](../study/plastic-projection/README.md)。

- [可塑方向基础课程评价](feedback-plasticity-course/README.md) → [冻结设计](../designs/feedback-plasticity-course.md)、[真实实施](../study/feedback-plasticity-course/README.md)、[固定数据](../data/feedback-plasticity-course/README.md)。

- [候选覆盖追查](candidate-label-coverage/README.md) → [旧课程结果](memory-update-learning/README.md)、[原始数据](../data/memory-update-learning/README.md)。

- [首步方向诊断](first-action-outcomes/README.md) → [设计](../designs/first-action-outcomes.md)、[原父轨迹](../data/window-cadence/README.md)、[分支数据](../data/first-action-outcomes/README.md)。

- [写入约束诊断](update-constraints/README.md) → [窗口原始数据](../data/window-cadence/README.md)。

- [后到归因评价](credit-history/README.md) → [设计](../designs/credit-history.md)、[数据](../data/credit-history/README.md)。

- [动作输入学习](action-update-learning/README.md) → [设计](../designs/action-update-learning.md)、[数据](../data/action-update-learning/README.md)。

- [候选动作影响](candidate-steering/README.md) → [窗口原始数据](../data/window-cadence/README.md)。

- [窗口比较](window-cadence/README.md) → [设计](../designs/window-cadence.md)、[数据](../data/window-cadence/README.md)。

- [负反馈提前判断](feedback-trigger/README.md) → [设计](../designs/feedback-trigger.md)、[数据](../data/feedback-trigger/README.md)。

- [受伤与写入时序](survival-timing/README.md) → [数据](../data/survival-feedback/README.md)、[此前结果](survival-feedback/README.md)。

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
