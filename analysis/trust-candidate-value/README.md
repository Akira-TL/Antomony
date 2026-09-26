# Analysis: 行动概率约束候选后果描述

## Navigation

- [研究首页](../../RESEARCH.md)
- [研究设计](../../designs/trust-candidate-value.md)
- [数据集](../../data/trust-candidate-value/README.md)

## Question / Target Contrast

同状态接受一次概率约束候选相对跳过，在后续最多64步内是否产生可区分的奖励、交付、伤害与死亡差。

## Inputs and Data Freeze

批次 `20260926T091027-2`，四世界58对，完整346文件清单及来源登记已提交 `af253b5`。本分析首次计算前登记计划，不重采样、不读取旧分析临时输出。

## Unit of Inference

两个整巢世界种子，条件配对，同世界候选相关。只逐世界描述，不把58点当独立重复，不计算显著性或总体置信区间。

## Primary Analysis

统一使用接受减跳过的差值。逐世界报告焦点奖励、焦点与整巢交付、焦点伤害和死亡差的正负、相同数量及均值、最小、最大；奖励和伤害数值容差1e-6，计数容差0。交付差正值有利，伤害和死亡差负值有利，不混淆方向。另报告参数实际改变数及已见窗口平均和最大状态相对熵。

## Exploratory / Sensitivity Analyses

不适用：不训练分类器、不事后筛选点、不改变窗口或扩大样本。

## Assumptions and Diagnostics

校验完整清单、来源模型、世界与候选身份、数量、维度、有限值、分支时限、无害条件无死亡、非零候选的已见概率约束及主轨迹残差快照零值。当前单次共同随机延续与短窗口不足以证明长期稳定增益或接受决策可学。

## Outputs

待运行：计划生成逐世界结构化描述，不提前写入结论。

## Reproduction

入口 `scripts/analyses/trust-candidate-value.sh`，实现 `scripts/analyses/trust_candidate_value.py`，独立配置 `.research/analysis/trust-candidate-value/A001/config.json`。输出使用独占创建，不覆盖已有结果。

## Result Boundary

只描述冻结开发条件下单次候选的短期后果。候选改善点数不能相加当成完整策略的搬运收益；过去替代目标改善不能替代未来改善。不是模型已学会控制更新，也不是未见环境优势或跨模型公平比较。

## Amendments

不适用：当前没有分析结果或结果后修订。
