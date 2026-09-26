# Study: 行动概率约束候选采样

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/trust-candidate-value.md)

## Study Identity

本地计算模拟，批次 `20260926T091027-2`，已正常完成一次固定配置采样。冻结提交 `56ff28d`，数据库登记成功并提交 `51cca61` 后才启动完整运行。

## Source and Experimental Units

整巢世界种子9801、9802，分别配对无害与持续伤害条件。稳定记录身份为 `benign-9801`、`benign-9802`、`persistent-9801`、`persistent-9802`。每巢八个独立模型，同巢个体与候选不是独立重复。

## Actual Groups / Exposure / Intervention

四条主轨迹全部跳过修改；每个入选点复制接受一次与跳过两个分支。实际点数依次16、16、13、13，共58。两个伤害世界各8只死亡，终止点未用下一代替代，未追加样本补满64点。

## Sample Collection and Processing

使用固定轮转规则选择非零存活候选，没有按未来结果挑选。零增量候选计数依次18、20、31、35，终止候选各8。原始行记录与压缩主轨迹直接保留，不进行结果筛选。

## Assays and Measurements

每分支最多64物理步，记录焦点及整巢后果、概率变化与替代目标变化。当前实施记录不汇总分支收益；分析另行登记。主轨迹每64步及结束保存参数与残差，共168组双文件快照。

## Protocol / Materials / Instruments

执行提交 `51cca61`，核心候选 `f89bd6c`，入口 `992fd14`；项目锁定的Python与PyTorch环境。协议 `.research/protocols/trust-candidate-value.json`，实际完整配置与输入散列在运行目录 `execution.json`。没有启动原V2V或浏览器。

## Batch / Run / Time

运行目录 `logs/trust-candidate-value/20260926T091027-2/`。协调世界步数依次281、277、296、331。元数据首写于2026-09-26 09:10:28 UTC，世界汇总末写于09:11:01 UTC；时间由文件记录定位，不用作精确性能测量。

## Failures / Missing Events

进程退出码0，未遇实现失败，输入文件散列检查通过。58点小于上限是按规则自然终止与候选资格所致，不视为丢失后填补。

## Deviations

不适用：本次采集未发现相对冻结设计的实施偏离。短流程9599在设计冻结前完成，仅检验程序，不混入此批数据。

## Outputs

四个 `pairs.jsonl`、四个 `parent.jsonl.gz`、执行元数据、世界记录及336个参数文件，共346个文件交给独立数据集清单。没有将未来分支回写在线模型。

## Record Boundary / Corrections

运行目录、点数、步数、计数和时间来自实际进程及文件记录；它们不是对候选有效性的解释。后续描述分析不能把这些主轨迹快照称为训练进展。
