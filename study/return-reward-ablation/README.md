# Study: 空载返巢奖励配对诊断

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/return-reward-ablation.md)

## Study Identity

身份 `return-reward-ablation`，本地连续模拟。2026-09-26 14:21:54.302021 UTC开始，当前执行中；尚未完成结果核对。

## Source and Experimental Units

世界种子19301、19302；每个世界32只，八份既有初始化各独立复制四份。实验单位为世界种子，样本键 `<variant>-moving-danger-<seed>-<arm>`，奖励配置、组及个体相关。

## Actual Groups / Exposure / Intervention

按冻结计划依次执行 `on` 与 `off` 两批，分别保留2分及取消空载返巢奖励。每批两个种子、学习接受、全跳过、固定接受、冻结MLP和纯规则五组。无新预训练、无新参数规模；危险与复活规则不变。

## Sample Collection and Processing

原始记录写入 `logs/return-reward-ablation/20260926T142153-2/` 的独立组目录，不覆盖8774旧数据。每步观察在动作前，事件与身体在动作后；所有记录保留，不按表现排除。

## Assays and Measurements

连续记录行动、实际拾取交付、体力与危险终止、复活、伤害、奖励及更新提案。每256步与终点保存参数和记忆。正在产生数据，尚未进行正式配对分析。

## Protocol / Materials / Instruments

设计冻结提交 `81ac692`，执行起点 `a8fc37b2f21cffcb7d80a5f278b304531884ec31`。配置为 `.research/protocols/return-reward-on.json` 与 `return-reward-off.json`；既有基础和接受参数由各批 `execution.json` 固定散列。执行期间只新增分析文件，不改变仿真或模型源代码。

## Batch / Run / Time

入口 `scripts/training/return-reward-ablation.sh`，批次 `20260926T142153-2`；最多8个独立进程，各进程PyTorch单线程。结束时间尚待执行器记录。两批启动时分别记录实际Git提交，后续分析开发提交不会解释为模型改变。

## Failures / Missing Events

当前无已确认失败；进程仍执行中，不能将尚未产生的世界视为缺失或完整。

## Deviations

当前未发现相对设计的实施偏离。

## Outputs

每批 `execution.json`、`worlds.jsonl`、逐世界 `result.json`、`trajectory.jsonl.gz`、`updates.jsonl` 及参数文件；执行输出另保存在批次根目录的两份日志。完成后生成完整散列清单再交给数据登记。

## Record Boundary / Corrections

开始时间及版本来自运行器实录；完成状态不提前填写。原工程种子19201只用于提出问题，不进入本次结果。
