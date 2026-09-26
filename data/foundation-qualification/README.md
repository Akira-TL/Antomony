# Dataset: 修正后三组基础轨迹

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/foundation-qualification/README.md)

## Dataset Identity

身份 `foundation-qualification`，版本 `20260926T104801-2`，48个世界记录及其完整轨迹。

## Research Purpose

描述修正规则后的基础搬运、探索和返回条件，识别能力差异。不是未知环境适应性收益数据。

## Source and Version

本项目固定模拟采样，执行提交 `c9ed693`。源模型身份和SHA-256存于批次 `sources.jsonl`，各个体模型来源不混并为同一训练实例。

## Population and Sample Mapping

按世界种子、任务、策略配对。种子14101至14108，任务为食物/无食物，策略为循环/MLP/规则。独立单位是世界，而非48条轨迹或384个个体实例。

## Data Layers and Artifacts

52份本地原始文件由 [manifest.sha256](manifest.sha256) 固定。原始目录 `logs/foundation-qualification/20260926T104801-2/`；分析结果另存，不覆盖轨迹。模型文件不复制，但来源清单保存17份固定参数身份与散列。

## Metadata / Missingness / Exclusions

没有按表现排除世界。无食物任务库存为0；结束原因可为时限或所有个体耗尽。未完成搬运不能把终止步数当完成时间。返回后耗尽可同时计数。

## QC and Anomalies

采样已检查零参数更新及源文件不变。分析将检查全部身份、来源散列、连续时间步、个体数量、携食变更重建的拾取与交付，以及从位置重算的最大半径；任一不符停止解释。

## Processing and Reproduction

无筛选或变换；原始文件逐份SHA-256。入口 `scripts/training/foundation-qualification.sh` 重建新批次，不能覆盖本版本。

## Freeze / Access / Ethics

完整清单与身份提交后供分析。纯模拟，文件只保存在本机，不访问V2V，不外传数据；本地冻结不等于外部预注册。
