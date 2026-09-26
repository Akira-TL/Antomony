# Dataset: 复杂来源连续轨迹与参数

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/continuous-adaptation/README.md)
- [冻结设计](../../designs/continuous-adaptation.md)

## Dataset Identity

身份 `continuous-adaptation`，版本 `20260926T123655-2`，80世界、6634文件。完整文件身份由 [manifest.sha256](manifest.sha256) 固定。

## Research Purpose

检验从基础参数出发的连续在线更新，是否在三种复杂来源下改善搬运与存活；主要区分学习接受相对全部跳过及固定接受，MLP与规则仅补充比较。

## Source and Version

由执行提交 `95b920f4d097cd355c8fe4745de0948318b66cb5` 产生，2026-09-26 12:43:57.902353 UTC完成。`execution.json` 固定协议散列及25个输入文件身份；没有数据原位修改。

## Population and Sample Mapping

样本键为 `<condition>-<seed>-<arm>`。四种子18101至18104，四条件 `reference/slow/periodic/moving-danger`，五组 `learned/skip/always/mlp/rules`。每样本八个体；个体索引0至7对应固定的独立模型。推断单位是世界种子，不把80世界、个体或时间步当作互相独立的重复。

## Data Layers and Artifacts

原始目录 `logs/continuous-adaptation/20260926T123655-2/`，约206 MB。根目录含执行身份及80条世界记录，各世界包含结果、压缩逐步轨迹、提案记录和定期参数。全部原始文件保持只读用途；后续派生分析另存。

## Metadata / Missingness / Exclusions

`Frame.tick` 是完成动作后的步数，来源位置及激活状态对应动作前的 `tick-1`。个体观察在动作前，位置和累计事件在动作后。死亡同时属于耗尽，携食状态可能在死亡后保留。终止后的累计结果不再增加；不伪造剩余时间的动作，不按存活时间加权。规则组没有模型参数文件，记录中的快照时刻仅是共同计划时刻。

## QC and Anomalies

已确认进程正常结束、80世界记录齐全；运行器检查基础参数冻结及25个源文件散列不变。完整轨迹重建、提案接受与实际写入对应、快照参数及外部干预一致性仍待正式分析审计。此前接受课程失败保留，不能把当前文件存在视为自训练有效。

## Processing and Reproduction

入口 `scripts/training/continuous-adaptation.sh`，配置 `.research/protocols/continuous-adaptation.json`。清单逐文件SHA-256，不按得分过滤。再次运行必须使用新目录。初始、每128步及终止参数可用于比较行为，但快照不是包含环境与随机状态的完整续跑存档。

## Freeze / Access / Ethics

本地模拟记录不强制加入Git，清单、协议及入口受版本控制；不外传、不接触原V2V。无真实受试者，内部冻结不称外部预注册。后续结论只能覆盖这批固定开发条件，尚非一般优势证明。
