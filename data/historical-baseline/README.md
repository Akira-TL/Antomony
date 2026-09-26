# Dataset: 历史奖励基线连续轨迹与参数

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/historical-baseline/README.md)

## Dataset Identity

`historical-baseline`，版本 `20260926T151701-2`。

## Research Purpose

检验仅改变历史基线是否值得进一步训练接受课程，不将软件测试代替真实结果。

## Source and Version

本机连续执行器产生，源码 `41b3d82`。原始目录每配置 `execution.json` 保存完整来源与时间，来源参数运行前后核对未变。

## Population and Sample Mapping

两个世界、两配置、五组，共20世界。样本身份为配置、条件、世界种子和组，实验单位为世界种子。个体和时间窗不当作独立样本。

## Data Layers and Artifacts

原始层 `logs/historical-baseline/20260926T151701-2/`；含逐帧轨迹、提案、全部定期快照、执行记录和日志。[完整清单](manifest.sha256)固定全部文件。派生结果另存，不回写原始层。

## Metadata / Missingness / Exclusions

无结果后排除，全部预定世界正常结束；真实负值、零交付和未接受提案均保留。

## QC and Anomalies

运行前后来源散列一致，完整的逐帧及基线递推核对待正式分析执行。

## Processing and Reproduction

入口 `scripts/training/historical-baseline.sh`，固定两配置；重复必须新建目录。分析从事件重建奖励与基线，不只读取终点汇总。

## Freeze / Access / Ethics

本地模拟，无真实受试者。大体积数据不提交Git，完整散列和来源登记入库。新种子不意味着新危险机制，后续不可再把本批称未见评价。
