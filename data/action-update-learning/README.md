# Dataset: 四步动作输入基础配对数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/action-update-learning/README.md)

## Dataset Identity

身份 `action-update-learning`，版本 `20260926T173720-2`，12世界、39配对，训练25、留出14。作为新旧接受输入比较的固定原始数据。

## Research Purpose

在完全相同的候选状态上比较参数输入与动作输入的基础接受学习，不作为生存目标或复杂环境训练数据。

## Source and Version

设计 `c9769ca`，执行 `bf3aec3`，UTC结束 `2026-09-26T17:38:28.620496+00:00`。原始目录 `logs/action-update-learning/20260926T173720-2/`，来源检查点与散列在 `execution.json`。

## Population and Sample Mapping

训练19801至19804、留出19811和19812。`initial-0` 为零偏移，`initial-1` 为0.75同向偏移；目录、种子、时点和焦点组成配对身份。独立单位为整巢种子，初始化条件配对。

## Data Layers and Artifacts

1306份原始文件由 [manifest.sha256](manifest.sha256) 固定。没有标签变换、样本清理或原地覆盖。后续接受参数和预测汇总另存分析输出。

## Metadata / Missingness / Exclusions

零候选/终止点不采配对，但世界数量和计数保留；19803偏移条件配对为0。缺少动作记录不得补零，少样本不得从别的个体或留出世界补齐。新动作字段含固定变换后的38维和版本。

## QC and Anomalies

采样退出成功、源模型及父轨迹残差不变、无危险死亡。完整清单、640组快照、4502帧、分区及动作输入格式在分析前继续核验；本记录不等同于分析审计或效果验证。

## Processing and Reproduction

入口 `scripts/training/action-update-learning.sh`；采样模块为 `candidate_probe.py`、`update_curriculum.py`。清单为原始文件SHA-256，不变换科学测量。不得以重新运行覆盖此批。

## Freeze / Access / Ethics

本记录、清单及登记提交后冻结为分析输入。大型本地轨迹与模型保持忽略，通过清单追踪；无个人资料、真实动物或外传。生存目标迁移仍未验证。
