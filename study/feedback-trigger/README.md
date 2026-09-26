# Study: 负反馈提前判断连续采样

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/feedback-trigger.md)

## Study Identity

`feedback-trigger`，固定连续模拟；批次 `20260926T163643-3`，16世界完成。

## Source and Experimental Units

19601、19602两个世界种子；样本由配置、条件、种子和组唯一标识。每世界8个相关个体。

## Actual Groups / Exposure / Intervention

固定窗口与负反馈提前触发，各含全部跳过、固定接受、普通MLP、纯规则。两配置均用生存反馈，不改惩罚和模型。

## Sample Collection and Processing

2048步固定时限，保留记忆复活；每配置4个隔离进程，配置依次执行。没有新增训练基础或接受模型。

## Assays and Measurements

完整轨迹、反馈和提案，每128步及终点参数、记忆与随机状态。没有全场中间状态快照。

## Protocol / Materials / Instruments

代码提交 `8ffe6bc75a9742b6f769788263529540abdee455`；执行入口 `scripts/training/feedback-trigger.sh`，外层 `timeout 300`。两份执行记录固定协议和源参数散列。

## Batch / Run / Time

UTC2026-09-26 16:36:44.120676至16:37:48.310718，约64秒；直接依据两份 `execution.json`。

## Failures / Missing Events

无执行失败，16世界均达到2048步；部分个体复活时等待巢内空位属于既有规则，不是缺失数据。

## Deviations

无已知实施偏离，仍须按正式分析核对完整记录。

## Outputs

`logs/feedback-trigger/20260926T163643-3/` 保存原始世界、执行身份与日志。

## Record Boundary / Corrections

执行后按直接日志整理，不修改冻结设计或原始结果。
