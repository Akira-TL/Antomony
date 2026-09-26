# Study: 生存反馈连续采样

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/survival-feedback.md)

## Study Identity

`survival-feedback`，固定连续模拟，16世界完成；实际批次 `20260926T161127-3`。

## Source and Experimental Units

世界种子19501、19502；样本身份为配置、条件、种子、组的组合。每世界8只共享场，个体不是独立实验重复。

## Actual Groups / Exposure / Intervention

旧反馈与生存反馈各执行全部跳过、固定接受、普通MLP、纯规则。危险源按预定路线移动；没有主动追逐，不训练旧接受器。

## Sample Collection and Processing

每世界固定2048步，死亡或耗尽保留参数和记忆复活。原始日志不筛选个体或删除失败；两配置串行，每配置最多4个隔离进程。

## Assays and Measurements

逐帧轨迹及提案；每128步与末尾参数、记忆、随机状态快照。未保存全世界场快照，不声称能从任意中间点无缝续跑。

## Protocol / Materials / Instruments

源提交 `0bc891a6d15214909730ea1bc23c4d14cfd2b78c`；两份 `survival-feedback-*.json` 协议散列及全部源模型散列由执行记录保存。通过项目 `.venv` 与 `uv` 执行；入口 `scripts/training/survival-feedback.sh`，外层 `timeout 300`。

## Batch / Run / Time

UTC起始2026-09-26 16:11:28.674764，结束16:12:31.876192，约63秒。两份 `execution.json` 是直接时间来源，未超300秒预算。

## Failures / Missing Events

未发生执行退出失败；两个配置各8个结果。科学效果另由分析判断，运行完成不代表采用成功。

## Deviations

无已知实施偏离。分析前仍须完成逐帧机械核验。

## Outputs

`logs/survival-feedback/20260926T161127-3/`，含两配置完整原始世界、执行记录和控制台日志。

## Record Boundary / Corrections

本记录在执行结束后根据上述直接日志整理，不冒充同时记录的人类观察；没有改写原始日志或冻结设计。
