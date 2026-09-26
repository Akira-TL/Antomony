# Research Tree

## Graph

```mermaid
flowchart TD
    N1["针对选定赛题构建可验证的数学模型<br/>objective · active"]
    N2["有限近期修改与局部后续反馈能否改善连续预测中的选择性回退<br/>question · open"]
    N3["小型神经预测器选择性回退的草案设计<br/>design · open"]
    N4["检验蚁群自主参数更新的可行性与未见环境适应能力<br/>objective · active"]
    N5["局部反馈能否支持学会接受有益更新并适应未见环境<br/>question · active"]
    N8["候选短期价值描述：危险分支未显示后果差异<br/>analysis · resolved"]
    N10["取消返巢奖励未通过采用条件<br/>analysis · resolved"]
    N12["伤害候选质量与接受判断均有不足<br/>analysis · resolved"]
    N14["历史基线未通过且奖励与交付排序相反<br/>analysis · resolved"]
    N15["势函数辅助奖励在有限回合和更新截断中何时保留原任务<br/>question · resolved"]
    N17["生存反馈相对不更新改善但未稳定优于旧反馈<br/>analysis · resolved"]
    N18["多数死亡前已写入但仍未形成稳定避险<br/>analysis · resolved"]
    subgraph G5_design["design siblings"]
        direction TB
        N6["蚁群自训练适应性对照草案<br/>design · active"]
        N7["陌生信号候选更新的同状态价值诊断<br/>design · resolved"]
        N9["空载返巢奖励是否干扰危险适应<br/>design · resolved"]
        N11["伤害后单次候选与接受判断<br/>design · resolved"]
        N13["历史奖励基线的单变量连续对照<br/>design · resolved"]
        N16["生存优先的反馈有限对照<br/>design · resolved"]
        N19["仅提前负反馈候选时机的有限对照<br/>design · resolved"]
    end
    N1 --> N2
    N2 --> N3
    N1 --> N4
    N4 --> N5
    N5 --> N6
    N5 --> N7
    N7 --> N8
    N5 --> N9
    N9 --> N10
    N5 --> N11
    N11 --> N12
    N5 --> N13
    N13 --> N14
    N5 --> N15
    N5 --> N16
    N16 --> N17
    N17 --> N18
    N5 --> N19
```

## Node Index

| Node | 类型 | 状态 | 科研对象 | 人类入口 |
| --- | --- | --- | --- | --- |
| N1 | objective | active | 针对选定赛题构建可验证的数学模型 | [打开](../RESEARCH.md) |
| N2 | question | open | 有限近期修改与局部后续反馈能否改善连续预测中的选择性回退 | [打开](../docs/competition/v2v-extraction-scope.md) |
| N3 | design | open | 小型神经预测器选择性回退的草案设计 | [打开](../designs/neural-readout-selective-rollback.md) |
| N4 | objective | active | 检验蚁群自主参数更新的可行性与未见环境适应能力 | [打开](../RESEARCH.md) |
| N5 | question | active | 局部反馈能否支持学会接受有益更新并适应未见环境 | [打开](../designs/ant-self-training-adaptation.md) |
| N6 | design | active | 蚁群自训练适应性对照草案 | [打开](../designs/ant-self-training-adaptation.md) |
| N7 | design | resolved | 陌生信号候选更新的同状态价值诊断 | [打开](../designs/candidate-update-value.md) |
| N8 | analysis | resolved | 候选短期价值描述：危险分支未显示后果差异 | [打开](../analysis/candidate-update-value/README.md) |
| N9 | design | resolved | 空载返巢奖励是否干扰危险适应 | [打开](../designs/return-reward-ablation.md) |
| N10 | analysis | resolved | 取消返巢奖励未通过采用条件 | [打开](../analysis/return-reward-ablation/README.md) |
| N11 | design | resolved | 伤害后单次候选与接受判断 | [打开](../designs/injury-candidate.md) |
| N12 | analysis | resolved | 伤害候选质量与接受判断均有不足 | [打开](../analysis/injury-candidate/README.md) |
| N13 | design | resolved | 历史奖励基线的单变量连续对照 | [打开](../designs/historical-baseline.md) |
| N14 | analysis | resolved | 历史基线未通过且奖励与交付排序相反 | [打开](../analysis/historical-baseline/README.md) |
| N15 | question | resolved | 势函数辅助奖励在有限回合和更新截断中何时保留原任务 | [打开](../docs/research/reward-shaping-boundaries.md) |
| N16 | design | resolved | 生存优先的反馈有限对照 | [打开](../designs/survival-feedback.md) |
| N17 | analysis | resolved | 生存反馈相对不更新改善但未稳定优于旧反馈 | [打开](../analysis/survival-feedback/README.md) |
| N18 | analysis | resolved | 多数死亡前已写入但仍未形成稳定避险 | [打开](../analysis/survival-timing/README.md) |
| N19 | design | resolved | 仅提前负反馈候选时机的有限对照 | [打开](../designs/feedback-trigger.md) |
