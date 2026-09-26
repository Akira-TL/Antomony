# Analysis: 四步与十六步窗口的配对结果

## Navigation

- [研究首页](../../RESEARCH.md)
- [设计](../../designs/window-cadence.md)
- [数据](../../data/window-cadence/README.md)

## Question / Target Contrast

四步固定接受相对十六步固定接受及全部跳过的逐世界失败次数差。失败包括危险死亡与非危险体力耗尽，不重复计数。

## Inputs and Data Freeze

批次 `20260926T165559-3` 的16世界全部纳入，输入完整清单和模型来源已固定，不按结果筛选。

## Unit of Inference

两个世界种子，不把相关的个体、生命、窗口或时间当作独立重复。

## Primary Analysis

逐种子计算四步减两个参照的失败差。两个种子对两个参照都不增加失败，且对每个参照至少一个种子严格减少，才进入新接受课程设计。减少搬运不单独否决，更多搬运也不能抵消生存恶化。

## Exploratory / Sensitivity Analyses

分别报告死亡、耗尽、交付、伤害、判断、写入和前进次数，检查是否伴随行动减少。没有额外窗口、阈值或学习率搜索。

## Assumptions and Diagnostics

完整散列、来源及协议比对，核对2048步时限、库存、伤害反馈、终止与复活、每次提案窗口、参数链及快照。不更新的三个对照跨窗口必须行为相同。时序审计复用已测试的逐帧重建，判据测试覆盖缺失、重复、分量不符和搬运不能抵消失败。

## Outputs

待执行登记后的正式分析。原始计数可见，不在审计前将其升级为完整结论。

## Reproduction

`scripts/analyses/window-cadence.sh`；配置 `.research/analysis/window-cadence/A001/config.json`，完整输出单独保存。

## Result Boundary

只回答当前八模型库及两个既有危险机制的新布局，不构成普遍优势或基础能力等效证明。固定接受不是模型学会控制接受；未来学习接受需要新课程和留出验证。

## Amendments

无判据变更。正式分析登记发生于原始计数可见之后、配对分析执行之前；设计及主要判据在采样前冻结。
