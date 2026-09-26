# Analysis: 历史奖励基线配对后果

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/historical-baseline.md)
- [输入数据](../../data/historical-baseline/README.md)

## Question / Target Contrast

两个新世界种子、8只及最多2048步的远距移动危险条件中，固定接受组开启历史奖励基线相对关闭时的逐世界实际交付、危险死亡及体力耗尽差。

## Inputs and Data Freeze

数据 `historical-baseline`，批次 `20260926T151701-2`。完整文件散列固定；原始运行汇总已可见，不称盲法。规则来自结果前设计，正式计算在分析登记之后。

## Unit of Inference

两个世界种子，同一模型库；配置、个体、组和窗口相关。不作独立个体推断，不声称一般泛化。

## Primary Analysis

核对单变量配置、模型来源、完整散列、复活、食物守恒、参数写入及快照。从每个体逐帧奖励、真实提案边界重建标量基线，检查所有快照及累计窗口数。三个无更新对照跨配置行为应逐帧一致。

差值为开启减关闭。逐世界报告五组交付、危险死亡、耗尽、奖励及写入差。主要固定接受组在两世界均不得少交付、多危险死亡或多耗尽，且至少一世界交付提高或死亡下降才值得继续。学习接受仅次要描述，不以其结果代替预定主判断。

## Exploratory / Sensitivity Analyses

描述历史基线范围和窗口数，保留总奖励与交付方向相反的情况，不增加参数搜索或重新定义成功。

## Assumptions and Diagnostics

数值快照允许1e-6基线重建误差，事件计数精确一致。累计窗口数只在真实提案处理且选项开启时增加，不能随推理或当前待定反馈提前改变。独立手算与损坏快照测试先通过。

## Outputs

尚未正式计算。结果计划存入 `summary.json`，完整输入与派生输出独立保存。

## Reproduction

`bash scripts/analyses/historical-baseline.sh`，配置 `.research/analysis/historical-baseline/A001/config.json`。先执行两配置的复活完整性检查，再运行配对及基线核对；不覆盖已有结果。

## Result Boundary

探索性新随机实例，不是未见危险机制或自训练控制已经学会的证明。新标量基线不等于学习更新规则，旧接受模型未重训。奖励改善不能代替搬运能力改善。

## Amendments

初始分析，无结果后改判据。
