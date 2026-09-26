# Study: 修正后三组基础轨迹采样

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/foundation-qualification.md)

## Study Identity

批次 `20260926T104801-2`，48个基础世界全部结束。设计冻结于 `5f1f2d3`，成功登记后提交 `c9ed693`，其后才正式运行。

## Source and Experimental Units

世界种子14101至14108；每世界各三组两任务。每巢八只，世界为推断单位，各个体不是独立研究重复。

## Actual Groups / Exposure / Intervention

循环模型、信号阶段MLP和修正后纯规则。神经组固定采样方向，规则使用自身随机探索；各组环境、库存、信息素及模型独立。使用有限巢穴源与局部平滑沉积，不施加危险、减速或周期变化。

## Sample Collection and Processing

按冻结顺序执行全部48世界，没有排除或补充种子；全部参数更新次数为零。源模型运行前后散列一致。采样输出中的逐世界成绩已随运行显示，后续汇总不称盲法分析。

## Assays and Measurements

逐步动作、位置、携食与奖励保存在48份压缩轨迹中。世界记录另含交付、拾取、返回、耗尽及最大半径。固定参数来源共17个文件，未训练，因此不重复生成相同参数快照。

## Protocol / Materials / Instruments

协议 `.research/protocols/foundation-qualification.json`，入口 `scripts/training/foundation-qualification.sh`，执行提交 `c9ed69381cbbcaecb956547695d42058caf97645`。项目锁定Python/PyTorch，数值线程数1。

## Batch / Run / Time

目录 `logs/foundation-qualification/20260926T104801-2/`。执行元数据记录开始为2026-09-26 10:48:02 UTC，结束10:49:28 UTC，约86秒；不是包含工程开发、协议编写与分析的总成本。

## Failures / Missing Events

进程退出0，48条世界记录，无失败世界。详细身份、轨迹长度与统计计数一致性由分析再核对。

## Deviations

不适用：未发现实施偏离。两只八步的14999短流程属于工程检查，不进入正式数据。参数和条件未随运行结果改变。

## Outputs

52份文件：48条压缩完整轨迹，及配置、执行元数据、模型来源散列和世界记录。数据清单单独固定。

## Record Boundary / Corrections

已完成采样不等于三组能力相同。冻结基础参数也不证明自学习模型的接受决策有效。本次数据不得用于反复挑选模型或扩大本次样本。
