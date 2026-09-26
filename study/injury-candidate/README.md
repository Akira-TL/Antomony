# Study: 真实伤害后的单次候选分支

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/injury-candidate.md)

## Study Identity

`injury-candidate`，批次 `20260926T145121-2`，已完成。

## Source and Experimental Units

来源为返巢奖励诊断中保留奖励的学习接受组，世界19301与19302。每世界16个不同个体的首次合格点；世界是推断单位，同世界点与两分支相关。

## Actual Groups / Exposure / Intervention

相同状态、随机流与记忆下，焦点个体接受一次原候选或跳过，之后全部个体参数冻结。其余个体保留父轨迹当步反馈处理完的参数。复活和记忆保留继续生效。

## Sample Collection and Processing

只按已发生的伤害、非零候选和预定顺序取样，不按未来效果筛选。19301重建至240步、19302至539步，各收足16对，未用完1536步上限。父观察、动作、事件和更新记录逐值对照原轨迹。

## Assays and Measurements

每分支最多128步，保存逐帧身体、奖励、事件、累积死亡、复活及原候选；每点保存全32只参数、记忆和随机状态。配对后果由独立分析脚本重建。

## Protocol / Materials / Instruments

配置 `.research/protocols/injury-candidate.json`，入口 `scripts/training/injury-candidate.sh`，PyTorch单线程。执行源码 `9211f0c87864e14b1af2af34da2271a11dd965ac`。无新训练、无参数回写。

## Batch / Run / Time

UTC 2026-09-26 14:51:27.778252 至14:56:15.998140，约288秒；实际时间及采样点见原始 `execution.json`。

## Failures / Missing Events

运行正常结束，两世界均收足；正式事件与快照完整性待分析检查，不以运行成功替代科学结论。

## Deviations

没有修改取样或后果范围。使用已见主轨迹，属于开发诊断，不是新的留出评价。

## Outputs

`logs/injury-candidate/20260926T145121-2/`；[数据登记](../../data/injury-candidate/README.md)。

## Record Boundary / Corrections

父状态保存的是原决定之后参数；接受/跳过分支先恢复该候选前参数，再单独施加干预。分支的零写入计数表示延续期间冻结，不否认初始单次干预。未来分支不得作为本模型训练标签。
