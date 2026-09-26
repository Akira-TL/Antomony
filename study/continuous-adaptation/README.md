# Study: 复杂来源连续更新对照

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/continuous-adaptation.md)
- [原始数据](../../data/continuous-adaptation/README.md)

## Study Identity

批次 `20260926T123655-2`，四个种子、四条件、五组，共80个世界全部执行完成。设计冻结提交 `626fd3c`，成功登记后的执行提交 `95b920f4d097cd355c8fe4745de0948318b66cb5`。

## Source and Experimental Units

世界种子18101至18104是独立推断单位，同种子的条件、模型组、同巢个体和时间步相关。每世界八只蚂蚁；仅使用一组八个基础初始化及对应独立接受模型，不代表多个独立训练批次。

## Actual Groups / Exposure / Intervention

学习接受、全部跳过、固定接受三组使用相同基础、记忆和动作参数与随机流，仅方向修正的接受方式不同；另含17宽普通MLP与纯规则。四条件为正常、持续减速、周期减速、预定路线移动混合危险来源。后三者从动作第128步生效，最多768步、库存48。各组环境和信息素独立，干预按种子和时间配对。有限巢穴源半径1.5，不给模型全局方向。

## Sample Collection and Processing

固定顺序完成80世界，没有追加种子、剔除失败、延长运行或重训接受模型。初始方向修正为零，在线仅依据已发生的16步反馈提出更新，不调用未来分支。接受模型仍是前次未通过开发条件的候选。逐世界结果运行时可见，后续分析不称盲法。

## Assays and Measurements

每步记录八个个体的局部观察、动作、位置、携食、预算、拾取、交付、返回、伤害、死亡、耗尽、奖励和真实写入数；每项提案记录特征、预测、接受与参数前后值。初始、每128步及终止保存参数。快照不包含完整环境及随机状态，不能声称任意时刻无缝恢复。

## Protocol / Materials / Instruments

协议 `.research/protocols/continuous-adaptation.json`，入口 `scripts/training/continuous-adaptation.sh`。25个源参数文件及其SHA-256在 `execution.json`，含动作1、基础8、接受8、近似规模MLP8。接受参数来自[登记的固定副本](../../data/memory-acceptance-models/README.md)，MLP来自[基础数据](../../data/matched-foundation/README.md)。

## Batch / Run / Time

开始 `2026-09-26T12:36:56.318399+00:00`，结束 `2026-09-26T12:43:57.902353+00:00`，约421.584秒。单进程本机运行，退出码0；恢复上下文后继续读取同一进程结果，没有重启实验。

## Failures / Missing Events

未发现执行中断；80份世界记录齐全。死亡、耗尽与提前结束是观测结果，不作为失效运行删除。轨迹一致性、提案写入链及冻结参数仍须分析审计，不能仅凭退出码断言全部有效。

## Deviations

未发现执行协议偏离；不将后续数据核验自动等同于不存在任何实现错误。

## Outputs

原始目录 `logs/continuous-adaptation/20260926T123655-2/`，6634文件，约206 MB。完整清单由数据对象固定。后续按冻结设计核对参考条件、干预前一致性、真实写入及配对行为差异；主动追逐、死亡继承和已训练记忆不在本批证据范围。

## Record Boundary / Corrections

2026-09-26在分析执行期间将原实施记录的合并章节拆为规范章节，并补充目录关系；没有更改来源、执行事实、原始数据或冻结设计。该记录只确认实施，效果判断由另行登记的分析承担。
