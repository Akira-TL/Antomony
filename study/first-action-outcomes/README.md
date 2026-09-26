# Study: 首次受伤前单次方向选择的32步后果

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/first-action-outcomes.md)

## Study Identity

`first-action-outcomes`，正式采样完成，8状态、128分支。设计冻结提交 `1ea3656`，采样执行提交 `cd8ba63`。

## Source and Experimental Units

固定原批次 `20260926T165559-3/four` 的19701、19702固定接受组，各前四不同个体首次受伤前状态；整巢种子为单位。

样本登记的 `source_identity` 使用“种子/组”的逻辑身份标识，不是文件系统路径。对应真实父目录分别为 `logs/window-cadence/20260926T165559-3/four/moving-danger-19701-always/` 与 `logs/window-cadence/20260926T165559-3/four/moving-danger-19702-always/`；轨迹、更新链及快照均从这两个实际目录由既有世界解析器读取，清单逐文件核验。

## Actual Groups / Exposure / Intervention

只在首步覆盖焦点方向，随后所有参数冻结；每状态16方向各32步，共4096分支世界步。

## Sample Collection and Processing

原世界从初始化顺序重建；逐帧对照现有记录，不补采。19701点为(182,3)、(220,0)、(459,1)、(607,6)；19702为(241,1)、(282,4)、(340,3)、(449,6)，括号为干预前步数与零起始个体编号。两父前缀608/450步、8464次实际动作、56份非零记忆及随机状态快照核验通过。

## Assays and Measurements

原始逐帧轨迹及点初始参数、隐藏状态、随机数、场；失败不与死亡重复相加。

## Protocol / Materials / Instruments

`.research/analysis/first-action-outcomes/A001/config.json`，`scripts/analyses/first-action-outcomes.sh sample`，项目锁定依赖，单线程。

## Batch / Run / Time

固定目录 `logs/first-action-outcomes/A001`。执行元数据UTC开始19:48:03.790570，完成19:48:40.417033；单调计时34.89秒，退出码0。时间来源不同保留原值，不反推替换。采样硬上限240秒；审计汇总最多60秒，总计算300秒。

## Failures / Missing Events

无缺失分支或正式运行失败；未重试。准备中的机械测试曾发现测试浮点严格相等和测试配置字段问题，均在正式冻结前修正，四项通过，不计为正式采样。

## Deviations

暂无；未来执行偏离须另记，不改冻结设计。

## Outputs

209份文件由数据清单固定；包括8份点记录、128份逐帧分支、64份初始推理状态、8份信息素场及1份执行信息。

## Record Boundary / Corrections

未来结果只能用于首步后果诊断，不用于训练或部署。
