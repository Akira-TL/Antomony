# Dataset: 伤害候选配对轨迹

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/injury-candidate/README.md)

## Dataset Identity

`injury-candidate`，版本 `20260926T145121-2`。

## Research Purpose

区分伤害后候选本身的后果与原接受判断，不检验一般优势。

## Source and Version

来源固定于 `data/return-reward-ablation/manifest.sha256`；执行前后核对24386个源文件，无源数据改写。新执行身份与配置见原始目录 `execution.json`。

## Population and Sample Mapping

两个世界各16个首次合格个体点，共32对64分支；每个点的两分支对应同一个样本，不视为独立个体。实际身份为世界、父步数、个体编号。

## Data Layers and Artifacts

原始层 `logs/injury-candidate/20260926T145121-2/` 含执行记录、每点全体参数与记忆、原决定、分支结果和完整压缩轨迹。[完整散列清单](manifest.sha256)固定全部文件；派生分析单独输出，不回写原始目录。

## Metadata / Missingness / Exclusions

两世界收足预定样本，没有结果后排除。未采样点不表示候选无效或有效。仅涵盖主轨迹早期受伤个体。

## QC and Anomalies

父轨迹重建逐值一致；正式分析核对3169个文件及8192帧的散列、食物库存、事件、累积死亡、复活与快照通过。全部分支完成128步；所有原始负后果及零差值保留。

## Processing and Reproduction

使用 `scripts/training/injury-candidate.sh` 和固定配置；新执行必须另建目录，不覆盖本数据。

## Freeze / Access / Ethics

本地模拟，不涉及真实受试者。原始大体积文件不提交Git，散列清单提交。既有留出数据已用于开发诊断，不再声称本批对后续新方案未见。
