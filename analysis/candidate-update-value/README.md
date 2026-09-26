# Analysis: 候选更新短期价值描述

## Navigation

- [Research](../../RESEARCH.md)
- [Design](../../designs/candidate-update-value.md)
- [Dataset](../../data/candidate-update-value/README.md)

## Question / Target Contrast

禁止更新主轨迹上接受一次45参数候选与跳过的后续最多64步焦点反馈和实际交付配对差。

## Inputs and Data Freeze

输入为批次 `20260926T081311-2` 的四个世界、72 对候选。清单 `data/candidate-update-value/manifest.sha256` 已提交，原始输入未改写。采样数量及主轨迹结束记录已可见，但首次汇总分支差值前登记本分析。

## Unit of Inference

两个世界种子，条件配对，候选为相关重复。只作逐世界描述，不按72个独立样本计算置信区间或显著性，也不声称跨世界泛化。

## Primary Analysis

按世界列焦点奖励差的平均、最小、最大、正负及相同数量，数值容差为 `1e-6`。同时列焦点和整巢交付改善/恶化点数、焦点死亡增减点数。方向统一为接受减跳过。候选的改善点数不能相加解释成完整策略可实现的搬运收益。

## Exploratory / Sensitivity Analyses

不适用：本次不训练分类器，不按事后收益重新筛选或扩张分组。

## Assumptions and Diagnostics

校验哈希、模型化字段、有限值、世界身份、候选唯一性、计数、时限及无害条件无死亡。单次共同随机延续可能产生偶然正负差，无法据此证明更新价值稳定或局部可预测。终止点缺少存活候选是设计采样范围，而不是填补为零标签。

## Outputs

尚未产生分析结果。计划生成逐世界机器表和一致性检查状态；完成后登记具体输出位置。

## Reproduction

入口 `scripts/analyses/candidate-value.sh`，配置 `.research/analysis/candidate-update-value/A001/config.json`；只消费已登记原始数据，不重新采样，不读取其他分析的临时目录。

## Result Boundary

只回答本候选在当前开发轨迹和时限内的后果。不是模型已经学会选择修改，不是未见复杂环境优势，也不能替代 MLP 和纯规则比较。

## Amendments

不适用：尚无修订。
