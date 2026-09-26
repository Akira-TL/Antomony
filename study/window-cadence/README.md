# Study: 四步与十六步窗口连续采样

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/window-cadence.md)

## Study Identity

`window-cadence`，固定连续模拟，批次 `20260926T165559-3`，16世界均已完成。

## Source and Experimental Units

19701、19702两个世界种子，每世界8个相关个体。样本由窗口配置、条件、种子和组唯一标识，独立单位为世界种子。

## Actual Groups / Exposure / Intervention

十六步与四步固定窗口，各包含全部跳过、固定接受、普通MLP及纯规则；两者都用生存反馈，没有逐负反馈提前触发，没有运行旧学习接受器。

## Sample Collection and Processing

每配置4个隔离进程，两个配置依次执行，固定2048步，死亡和耗尽保留记忆复活。未续训基础或接受模型，未回写源参数。

## Assays and Measurements

完整行为、反馈及候选记录；每128步和终点保存参数、记忆和随机状态。各样本对应同一连续轨迹测量，没有中间全场状态快照。

## Protocol / Materials / Instruments

执行代码 `441098012df5b2d174e2cbc31065c4ec1bdbf257`，入口 `scripts/training/window-cadence.sh`，外层 `timeout 300`。运行期间仅新增尚未执行的分析文件，没有改动采样模块；源模型、展开配置及散列在两份执行记录中固定。

## Batch / Run / Time

UTC2026-09-26 16:56:00.452838至16:57:05.295774，约65秒；依据实际两份 `execution.json`。四步配置16:56:32.143367开始，前一配置16:56:31.188066结束。

## Failures / Missing Events

无执行失败或超时，16世界均2048步、16384有效个体步。没有删除失败生命或零搬运世界。

## Deviations

无已知采样偏离。原始计数已在正式分析登记前可见，分析不宣称结果盲化或外部预注册。

## Outputs

`logs/window-cadence/20260926T165559-3/` 保存所有原始产物及日志，由数据清单固定。

## Record Boundary / Corrections

本记录由执行结束后的直接日志整理；不以当前解释修改原始数据或冻结设计。
