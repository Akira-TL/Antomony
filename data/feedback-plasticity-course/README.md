# Dataset: 可塑方向基础课程原始记录

## Navigation

- [研究首页](../../RESEARCH.md)
- [真实实施](../../study/feedback-plasticity-course/README.md)
- [冻结设计](../../designs/feedback-plasticity-course.md)
- [配对评价结果](../../analysis/feedback-plasticity-course/README.md)

## Dataset Identity

数据集`feedback-plasticity-course`，版本`20260926T204123-2`，原始层，状态`active`。完整运行目录共129项文件，固定清单为本目录`inputs.sha256`，清单自身SHA-256为`a3d0030d1edb0316e495e0c9fdc70c5741bad4ff9fc210b2b22bcb37becca417`。

## Research Purpose

供固定基础课程的配对效果分析使用。数据身份核验不等于学习结果成立；当前已完成分析24，不按结果剔除回合，全部负结果保留。

## Source and Version

来源为Study 20，运行目录`logs/feedback-plasticity/20260926T204123-2/`。代码提交`5c9a338603dac0c206b2c74fb2ab5212b4769e36`，设计及协议冻结于`9aded454553027378580f2034182d3b0ef55fba2`。父Agent报告进程退出码0，完成记录声明两初始化完成、内部用时125.325716546秒；完成文件修改时间UTC 2026-09-26T20:43:37.447713705Z只作为完成记录时间，不冒充精确进程退出时间。

## Population and Sample Mapping

目录`seed-1701`与`seed-1702`分别对应两个外层初始化。每初始化150训练轮，各批16回合、96帧；评价每条件4批64回合。评价文件由条件、生成种子和模式标识，三个条件和四模式的对应关系由冻结协议定义。独立单位是外层初始化；回合、时点及配对模式不可当独立模型。

## Data Layers and Artifacts

原始目录保持不变，包含实际协议、代码/冻结身份、完成记录、训练逐轮日志、每初始化7个结构保存点和6个训练片段，以及全部评价轨迹和评价记录；两初始化共14个结构保存点和12个训练片段。未将这些原件复制成另一版本，也未在其上覆盖处理。清单进入Git，体积较大的原始文件在本地忽略目录保留，并在数据库以绝对位置登记。

运行控制台日志`logs/feedback-plasticity/20260926T204123-2.log`位于原始目录之外，由Study保存实施记录；129项清单的范围严格限于上述运行目录，不假称其包含目录外日志。

## Metadata / Missingness / Exclusions

协议副本与冻结协议逐字节相同，`commits.txt`已核对。清单生成记录、129项计数和逐项散列检查确认输入身份；分析24的唯一执行另已核验各组合、形状、保存点及内容契约。未根据奖励、接受率或成功与否排除任何记录，没有补采。训练片段按设计只每25轮保存，其缺少其他轮的完整帧不是结果后删除。

## QC and Anomalies

全部129项原始文件的SHA-256对固定清单校验通过，清单散列与父Agent固定值相同。分析24对147456评价帧的既定内容与配对检查通过，结果见关联正文；这些机械检查不等于课程通过，两个初始化均未达到预定效果门槛。若后续发现非有限值、配对缺失、来源变化、冻结参数改变或摘要不一致，保留该版本并停止有效性判断，不静默修原件或重新生成清单。

## Processing and Reproduction

原始记录由`scripts/training/feedback-plasticity.sh`产生。父Agent在进程结束且不再写入原始目录后，通过分析入口的`--freeze-inputs`独立模式固定清单，未在该步骤汇总效果；随后本任务以`sha256sum --check --quiet data/feedback-plasticity-course/inputs.sha256`核验。后续Analysis必须读取这份既有清单，不能在首次结果汇总时边计算边替换输入身份。

## Freeze / Access / Ethics

设计与预分析正文先于正式实施固定；本数据清单在首次效果汇总前固定，二者时间含义不同。纯本地合成方向任务，无真实生物或个人数据；原V2V只读、8775未改。数据可用于本固定课程的探索性评价，不能自行升级为真实蚁群搬运、生存或未见环境收益证据。
