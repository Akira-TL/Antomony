# Study: 无危险基础更新课程采样

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/basic-update-learning.md)

## Study Identity

批次 `20260926T094431-2`，固定采样正常完成。设计冻结提交 `310cbf4`，数据库登记成功并提交 `e927445` 后启动。

## Source and Experimental Units

八个整巢种子11101至11106、11111、11112；两种初始化条件。每巢八个独立模型。同巢个体与候选不是独立重复。

## Actual Groups / Exposure / Intervention

初始残差范数0和0.75，所有主轨迹跳过修改。16个世界均完成16次交付且零死亡。采集225对候选，训练170对、留出55对，不追加至256。

## Sample Collection and Processing

按窗口轮转选择非零存活候选，不按未来结果筛选。每点复制接受一次和跳过两分支。保留零候选、终止候选计数以及所有行记录，具体身份和数量见原始 `worlds.jsonl`。

## Assays and Measurements

分支最多64步；记录局部观察、记忆、候选、已见诊断与后续奖励及交付。主轨迹固定间隔和结束快照696组，非零初始化另含第0步。此处不计算分支收益或接受模型效果。

## Protocol / Materials / Instruments

执行提交 `e927445`，配置 `.research/protocols/basic-update-learning.json`；项目锁定Python/PyTorch环境，单线程数值执行。八个源模型及动作模型的散列记录于执行元数据，结束检查未改变。

## Batch / Run / Time

运行目录 `logs/basic-update-learning/20260926T094431-2/`。元数据写入09:44:31 UTC，世界记录最后写入09:47:32 UTC，来自文件时间，不作为精密性能计时。

## Failures / Missing Events

退出码0，无运行失败。自然终止导致部分世界少于16点，按原规则保留实际样本量，没有结果后填补。

## Deviations

不适用：未发现相对冻结设计的实施偏离。9599/9600短流程仅验收实现，不进入本数据。当前只完成采样，接受模型训练和留出计算尚未发生。

## Outputs

1426份原始文件，包括执行元数据、16条世界记录所在文件、16份候选、16份压缩轨迹、1392份模型与残差快照。全体交给数据清单，不覆盖或删选。

## Record Boundary / Corrections

交付数属于全部跳过的主轨迹表现，不能据此声称接受学习有效。候选未来不进入主轨迹。
