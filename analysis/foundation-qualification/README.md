# Analysis: 三组基础能力的配对描述

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/foundation-qualification.md)
- [原始数据](../../data/foundation-qualification/README.md)

## Question / Target Contrast

固定512步内三组在食物与无食物世界的交付、预算返回、耗尽和探索范围及逐世界配对差。

## Inputs and Data Freeze

版本 `20260926T104801-2`，48世界、52原始文件，身份与完整清单固定于 `d643ffa`。采样时逐世界结果已经显示；此处在首次执行汇总代码前登记，不宣称分析者未见原始结果。阈值与比较组均在采样前冻结。

## Unit of Inference

八个整巢种子，各组与任务配对；同巢个体及时间步不作为独立重复。

## Primary Analysis

首先验证全目录散列、实际协议、17个源模型、48个唯一世界身份、零训练、轨迹连续性与长度；以携食状态变化重建拾取/交付，以逐步位置重算最大半径。任一不符则停止。

各组按八个世界等权报告交付比例、无食物返回和耗尽比例、最大半径的均值与范围，附全部逐世界记录。搬运完成时间只描述确实完成的世界，并明示完成数量；配对时间差仅用于双方均完成的世界，同时保留全部世界的交付差。

循环组减MLP、循环组减规则的配对差保留负值。不做显著性或等效检验。依冻结设计计算三个开发门槛：平均交付至少0.9、平均无食物返回比例至少0.5、平均最大探索半径至少4.5；全通过也不能据此声明统计等效。

## Exploratory / Sensitivity Analyses

展示基础模型参数量与记忆，披露自学习部署额外45个方向参数和141个接受参数。没有追加种子、窗口或阈值搜索。

## Assumptions and Diagnostics

当前无死亡清除携食机制，携食真转假对应交付；原始环境代码与机械测试支持这一重建。返回与耗尽不是互斥事件。最大半径不等于面积覆盖；八个世界范围不代表总体置信区间。

## Outputs

待首次运行；生成逐世界记录、各组描述、配对差与开发门槛判断，不训练或选择检查点。

## Reproduction

入口 `scripts/analyses/foundation-qualification.sh`；代码 `scripts/analyses/foundation_qualification.py`；配置 `.research/analysis/foundation-qualification/A001/config.json`。结果独占创建，不覆盖已有尝试。

## Result Boundary

回答当前基础差异，不直接检验自主参数更新或未见复杂环境适应。即使基础搬运全部成功，探索差异仍可能阻止“同能力”的解释。

## Amendments

不适用：首次汇总尚未运行。
