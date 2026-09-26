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
    subgraph G5_design["design siblings"]
        direction TB
        N6["蚁群自训练适应性对照草案<br/>design · active"]
        N7["陌生信号候选更新的同状态价值诊断<br/>design · resolved"]
        N9["空载返巢奖励是否干扰危险适应<br/>design · resolved"]
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
