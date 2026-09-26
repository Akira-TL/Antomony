# Dataset: 四步判断下的动作保留配对数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/credit-history/README.md)

## Dataset Identity

身份 `credit-history`，版本 `20260926T180337-2`；四世界108条记录，对应两个种子的54个共同状态。

## Research Purpose

只比较候选更新的短期真实价值，不训练接受模型，不作为生存或陌生复杂环境的成功证据。

## Source and Version

设计 `7b1f8b8`，采样代码 `183da8a`。原始目录 `logs/credit-history/20260926T180337-2/`；源模型散列及协议见各配置的执行元数据。退出时间缺失及末次写入参考见实施记录。

## Population and Sample Mapping

目录 `four`、`sixteen` 对应两个动作保留范围，均四步判断；下级 `benign-19901`、`benign-19902` 固定种子。候选身份为种子、时点、焦点。整巢种子为独立单位，配置及时间点配对。

## Data Layers and Artifacts

556份原始文件由 [完整清单](manifest.sha256) 固定。保留全部零候选，无清理、补样或原地变换；分析输出另存。

## Metadata / Missingness / Exclusions

自然结束产生每配置30和24条，未达到每世界32条上限。终止候选不做分支。进程退出码未取回；缺失详情保留，不删除世界或重采。

## QC and Anomalies

四份世界表身份与数量齐全、均无死亡；1758帧、272组参数、源模型、文件完整性和两配置配对相同性须在分析中核验。任何核验失败时不输出有效效果结论。

## Processing and Reproduction

入口 `scripts/training/credit-history.sh`，模块 `candidate_probe.py`。清单使用SHA-256，原件不覆盖，不为了分析而重跑采样。

## Freeze / Access / Ethics

本记录、清单及数据库登记形成固定分析输入。大型轨迹与参数仅本地忽略保存；没有真实受试者、个人资料或外传。统计范围仅为当前两个开发种子。
