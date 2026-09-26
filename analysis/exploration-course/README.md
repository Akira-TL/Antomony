# Analysis: 无线索课程最终模型的配对结果

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/exploration-course.md)
- [原始数据](../../data/exploration-course/README.md)

## Question / Target Contrast

相同新评价世界中，32回合课程后的循环模型相对原模型的无食物最大探索半径、返回与耗尽差，以及食物交付与完成时间差。

## Inputs and Data Freeze

批次 `20260926T110225-2`，245文件清单与数据身份固定于 `255beec`。执行中逐回合原始结果已显示，分析不称盲法；组别、最终轮次和采用条件在正式训练前固定。

## Unit of Inference

八个整巢评价种子，条件于这一次训练所得的八模型。训练回合有参数继承，不能作为独立重复；不推断所有基础初始化。

## Primary Analysis

验证清单覆盖全部文件、协议一致、17个源文件未变、训练与评价种子及顺序、累计更新数不回退。逐轨迹重建交付及最大半径，冻结评价的更新数必须为零。训练轨迹允许真实累计更新。

逐个核对第0至32回合每4回合的72份模型和对应优化器：初始权重等于登记来源、快照更新数与训练记录一致、预留连接为零、权重有限、优化器学习率及衰减保持原值。参数变化范数仅说明发生更新，不作为能力指标。

报告四组逐世界成绩与世界等权均值/范围。主要对比为课程后减课程前的交付、无食物返回、耗尽与探索半径；完成步数只用于双方均完成的配对，并同时报告全部完成率。配对差保留正负，不做显著性或等效声明。

按冻结设计采用最终模型须同时满足：交付比例至少0.9，返回比例至少0.5且相对课程前下降不超过0.05，探索半径至少4.5。失败则不替换底座、不继续这套课程，不挑中间模型。

## Exploratory / Sensitivity Analyses

不新增中间轮次评价或奖励系数搜索。MLP、规则只提供同评价世界的能力背景；不能把课程训练与奖励重加权混成已学会自主参数更新。

## Assumptions and Diagnostics

轨迹重建依赖当前无死亡丢弃食物的基础环境。返回与耗尽可重叠，半径不等于覆盖率。观察到的结果可能受课程、奖励权重或现有学习器影响，本单方案不独立区分原因。

## Outputs

待首次汇总。输出完整评价记录、配对差、每个体参数变化及采用条件，不覆盖模型。

## Reproduction

入口 `scripts/analyses/exploration-course.sh`，代码 `scripts/analyses/exploration_course.py`，配置 `.research/analysis/exploration-course/A001/config.json`。共用轨迹核对位于 `src/mathhackson/training/comparison/auditing.py`，不依赖其他分析工作目录。

## Result Boundary

本次是基础能力补课，不是未知环境自训练检验。即使通过也须重新验证接受模型与新底座组合，不可自动沿用旧接受模型效能结论。

## Amendments

不适用：首次汇总尚未执行。
