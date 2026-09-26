# Dataset: 无线索探索课程与评价轨迹

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/exploration-course/README.md)

## Dataset Identity

身份 `exploration-course`，版本 `20260926T110225-2`，32训练回合、64评价世界和72组定期快照。

## Research Purpose

判断固定32回合课程是否扩大探索且保留返回与搬运，不通过本数据宣称未知环境自训练优势。

## Source and Version

执行代码 `dd8d596`，协议与元数据位于本批目录。原始参数仍在既有目录，17份来源身份与散列由 `sources.jsonl` 保存，未覆盖。

## Population and Sample Mapping

训练回合按种子15201至15232及连续回合编号对应；权重继承，非独立重复。评价按15101至15108、四策略和两任务配对，推断单位为条件于本次训练的整巢世界。

## Data Layers and Artifacts

245份原始文件全部由 [manifest.sha256](manifest.sha256) 固定，目录 `logs/exploration-course/20260926T110225-2/`。参数与优化器以回合、个体编号配对，144个文件对应72组，不误计为144个模型。

## Metadata / Missingness / Exclusions

没有按表现排除或换用中间快照。模型累计更新次数随个体活动时间不同。评价未完成搬运者保留，不能把终止时间当作完成时间；返回后耗尽可同时计入两项。

## QC and Anomalies

采样已检查冻结动作、零预留参数和来源不变。分析另核验245文件、32+64身份、轨迹时序、交互/半径重建、72组快照与更新数、零初始差异及预留连接。任何不一致先停止解释。

## Processing and Reproduction

无筛选和变换，逐文件计算SHA-256。运行入口 `scripts/training/exploration-course.sh` 只能创建新批次，不覆盖该版本。分析输出独立保存。

## Freeze / Access / Ethics

清单提交后作为固定分析输入。纯本地模拟，不读取受限数据，不外传、不修改原V2V。快照不含完整世界与随机流，不支持任意时刻精确续跑。
