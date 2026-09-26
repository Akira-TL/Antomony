# Dataset: 候选更新短期价值

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/candidate-update-value/README.md)

## Dataset Identity

`candidate-update-value`，单次固定开发采样的原始数据，非确认泛化测试集。

## Research Purpose

检验既有 45 参数候选的短期后果是否与全部跳过不同，为是否训练接受决策提供依据。不能直接证明门控可学。

## Source and Version

本项目采样批次 `20260926T081311-2`，源码提交 `790232d`，2026-09-26 接收。来源及输入模型哈希见原始 `execution.json`；分析输入文件哈希另存[清单](manifest.sha256)。

## Population and Sample Mapping

种子 9501、9502，各有 `benign`、`persistent` 两种条件。`condition + seed + tick + focal` 为候选唯一键，每条记录包含两个分支。两个独立世界种子，72 个相关决策点，不把候选当 72 个独立重复。

## Data Layers and Artifacts

原始数据位于 `logs/candidate-value/20260926T081311-2/`，不覆盖。`execution.json` 为元数据，`worlds.jsonl` 为采样计数，四份 `pairs.jsonl` 为配对结果，四份 `parent.jsonl.gz` 为主轨迹动作。参数快照同目录保留；当前没有筛选或衍生数据层。

## Metadata / Missingness / Exclusions

候选上限不是必须达到的样本量。按结果前规则排除终止和零增量，计数在世界记录中；没有事后排除。分支自然提前结束记录实际步数。终止继承不属于此数据的估计目标。

## QC and Anomalies

完整采样正常退出，源检查点未变。描述分析已核验清单哈希、字段、有限值、候选唯一性、32 点/世界上限、世界清单与记录数一致、合法时限及无害死亡为零，未发现异常；不作事后排除。

## Processing and Reproduction

采样入口 `scripts/training/candidate-value.sh`，生产代码 `src/mathhackson/training/foraging/candidate_probe.py` 和 `candidate_value.py`。没有原地变换；清单仅保存输入哈希。固定环境和九份模型均须保留才能重新采样，只有 Git 仓库不足以恢复本地权重。

## Freeze / Access / Ethics

当前快照由清单固定；大型原始数据和模型保存在忽略目录，路径及哈希登记但不公开上传。纯计算模拟，无真实动物或人类资料。普通主轨迹参数快照为全部跳过，不代表发生训练。
