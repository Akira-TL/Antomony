# Dataset: 近似规模模型及基础轨迹

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/matched-foundation/README.md)

## Dataset Identity

身份 `matched-foundation`，版本 `20260926T121232-2`，104份参数、96优化器状态与48世界完整轨迹。

## Research Purpose

核对固定预算训练后的普通MLP基础资格，为后续近似规模的完整复杂环境比较提供模型与基础差异，不直接评价在线自训练。

## Source and Version

由本项目执行提交 `e6e36ac` 产生，2026-09-26 12:14:15.961149 UTC完成。源文件九份的路径和SHA-256保存在原始目录 `sources.jsonl`。

## Population and Sample Mapping

世界17101至17108，每世界三组、食物/空场两任务。训练模型81至88映射为个体0至7；两神经组使用对应个体的同采样种子。训练个体不是48世界外的独立效果重复；本次仅一组八个初始化模型。

## Data Layers and Artifacts

原始目录 `logs/matched-foundation/20260926T121232-2/`，253文件由 [manifest.sha256](manifest.sha256) 固定。配置、执行时间、模型来源、96条训练诊断、世界记录及完整轨迹未做筛选；派生分析另存。

## Metadata / Missingness / Exclusions

没有按得分排除世界；无食物任务只清库存，返回后再耗尽同时计数。未完成任务的终止步数不当作完成时间。参数快照未包含完整环境与运行随机状态，不能当成任意时刻无缝恢复的运行存档。

## QC and Anomalies

执行期间核对全部评价零更新、源参数与最终文件未变；253文件清单完整。后续分析还须逐一核对参数宽度、初始权重、预留连接、冻结价值输出、优化器步数、世界身份和轨迹重建。不能用文件存在替代这些检查。

## Processing and Reproduction

入口 `scripts/training/matched-foundation.sh`；冻结配置 `.research/protocols/matched-foundation.json`，固定第2400轮作为最终模型。原始文件逐份计算散列，没有原位清洗或删除失败样本；新运行须写新目录。

## Freeze / Access / Ethics

本地模拟数据及模型不强制加入Git；清单、协议与入口跟踪版本，供后续分析和模型复用。无真实受试者，不外传文件，不接触原V2V，内部冻结不称外部预注册。
