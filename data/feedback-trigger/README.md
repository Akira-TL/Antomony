# Dataset: 负反馈提前判断轨迹与参数

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/feedback-trigger/README.md)

## Dataset Identity

`feedback-trigger`，版本 `20260926T163643-3`，全部16世界。

## Research Purpose

检验提前提出候选是否减少真实失败，而非把更多写入当学习有效。

## Source and Version

本机代码 `8ffe6bc` 生成，接收时间2026-09-26T16:37:48.310718+00:00，源模型身份由两份执行记录固定。

## Population and Sample Mapping

两世界种子各两配置四组，配置与组内个体、生命和时间相关；单位为世界种子。

## Data Layers and Artifacts

原始数据 `logs/feedback-trigger/20260926T163643-3/`，所有文件由[清单](manifest.sha256)固定。分析另存，不覆盖原始数据。

## Metadata / Missingness / Exclusions

无剔除。两配置均存真实学习反馈、新增伤害和原始环境奖励；复活等待是有效状态，不按失败后缺失处理。

## QC and Anomalies

正式分析强制核验清单、协议、计数、写入及触发时序；原始世界计数已可见，不把这次登记称为结果不可见的预注册。

## Processing and Reproduction

`scripts/training/feedback-trigger.sh` 生成，完整散列清单及人类来源先提交，再执行正式审计。

## Freeze / Access / Ethics

本机模拟，无受试者。原始大型文件不进Git，清单、配置和来源提交；不外传。
