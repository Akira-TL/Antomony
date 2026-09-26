# Study: 历史奖励基线连续采样

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/historical-baseline.md)

## Study Identity

`historical-baseline`，批次 `20260926T151701-2`，20世界完成。

## Source and Experimental Units

八份既有初始化模型，世界19401与19402；各配置及五组配对，推断单位为世界，不是个体或反馈窗口。

## Actual Groups / Exposure / Intervention

关闭基线与速率0.1，两配置除该选项外相同。五组均独立仿真；固定接受为主要对照，旧学习接受仅次要描述，无重新预训练。

## Sample Collection and Processing

8只、远距食物8至10、96份库存、2048步，复活保留记忆。两配置依次运行，各4进程，进程内PyTorch单线程，不共享模型存储。

## Assays and Measurements

逐帧轨迹、每次提案/决定、每128步及终点参数、历史基线、记忆与随机流。真实执行记录保存源参数散列。

## Protocol / Materials / Instruments

两配置 `.research/protocols/historical-baseline-{off,on}.json`；入口 `scripts/training/historical-baseline.sh`，源码 `41b3d8216e659a5b784398f8c91dfaeee6cb4a51`。执行期间分析代码新增，不改变两配置执行源码或输入。

## Batch / Run / Time

关闭：UTC 2026-09-26 15:17:02.084661至15:17:45.726839；开启：15:17:46.657760至15:18:30.546766。整体约88秒。

## Failures / Missing Events

20世界正常完成，均2048步；正式核对6146文件、40960帧及全部提案、记忆和基线快照通过，无排除。不以检查通过替代效果判断。

## Deviations

无执行偏离。原始汇总可见，正式分析之前登记规则，不称盲法。

## Outputs

`logs/historical-baseline/20260926T151701-2/`；[数据](../../data/historical-baseline/README.md)。原始文件不覆盖。

## Record Boundary / Corrections

8只与2048步只为本次配对筛选，不能与旧32只4096步直接比较。未接入8774，未从结果挑选参数。
