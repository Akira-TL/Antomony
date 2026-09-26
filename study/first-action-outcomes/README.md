# Study: 首次受伤前单次方向选择的32步后果

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/first-action-outcomes.md)

## Study Identity

`first-action-outcomes`，实施准备中，尚未生成正式分支。

## Source and Experimental Units

固定原批次 `20260926T165559-3/four` 的19701、19702固定接受组，各前四不同个体首次受伤前状态；整巢种子为单位。

## Actual Groups / Exposure / Intervention

只在首步覆盖焦点方向，随后所有参数冻结；计划16方向各32步，尚未执行。

## Sample Collection and Processing

原世界从初始化顺序重建；逐帧对照现有记录，按冻结规则取点，不补采。

## Assays and Measurements

原始逐帧轨迹及点初始参数、隐藏状态、随机数、场；失败不与死亡重复相加。

## Protocol / Materials / Instruments

`.research/analysis/first-action-outcomes/A001/config.json`，`scripts/analyses/first-action-outcomes.sh sample`，项目锁定依赖，单线程。

## Batch / Run / Time

计划固定目录 `logs/first-action-outcomes/A001`，采样硬上限240秒；审计汇总最多60秒。

## Failures / Missing Events

尚未执行，无结果；不得将准备中的目录当已接收数据。

## Deviations

暂无；未来执行偏离须另记，不改冻结设计。

## Outputs

待实际生成后固定文件清单并登记数据。

## Record Boundary / Corrections

未来结果只能用于首步后果诊断，不用于训练或部署。
