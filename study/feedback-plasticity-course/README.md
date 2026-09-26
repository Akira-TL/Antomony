# Study: 直接行动反馈驱动的可塑方向基础课程实施

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/feedback-plasticity-course.md)

## Study Identity

正式实施`feedback-plasticity-course`，状态`in_progress`。父Agent报告已于UTC 2026-09-26T20:41:23Z真实启动，进程会话22351，运行目录`logs/feedback-plasticity/20260926T204123-2/`。本次登记不是预定开始，亦不表示已完成或通过。结束时间尚未取得。

时序说明：此结构化开始状态在进程结束通知之后补记，先保存真实开始事件，再由独立记录推进完成；下文未取得完成信息的措辞描述启动时状态，不声称登记发生在训练结束之前，也不声称当前进程仍运行。

## Source and Experimental Units

协议固定外层初始化1701和1702，分别使用模型包内记忆底座00和01，共用动作底座。三个源文件路径及SHA-256由固定协议保存，冻结前已与`models/interactive/manifest.json`和实际文件核对。独立单位为外层初始化；其内部回合、条件和写入干预相关，不当作独立模型重复。

## Actual Groups / Exposure / Intervention

已启动的唯一协议包含原方向参照、持续关联、短暂反馈变化三个条件，以及学得接受、全部跳过、全部接受、等次数随机接受四组。固定每初始化150训练轮、每评价条件64回合。当前登记不读取评价内容，不声称某个组已经完成或获得改善；实际完成数量待进程结束后依据记录核验。

## Sample Collection and Processing

这是合成方向反馈课程，没有真实生物样本、物理位置、食物运输或死亡。程序逐回合产生局部线索，在线状态与随机流隔离，训练与评价生成种子分离。当前只登记真实启动和来源身份，不提前创建已采样、已完成的Sample或Assay，也不建立Dataset。

## Assays and Measurements

记录范围由冻结设计规定：训练每轮指标、每25轮结构参数与选定训练轨迹、全部评价逐帧输入/选择/反馈/在线矩阵/资格迹。当前尚未读取这些结果或进行汇总，不能从文件可能存在推断有效性或效果。

## Protocol / Materials / Instruments

冻结提交为`9aded454553027378580f2034182d3b0ef55fba2`；实际代码提交为`5c9a338603dac0c206b2c74fb2ab5212b4769e36`，均由运行目录`commits.txt`核对。运行目录`protocol.json`与`.research/protocols/feedback-plasticity-course.json`逐字节一致。入口为`scripts/training/feedback-plasticity.sh`，使用项目uv环境、单线程CPU和900秒外层强制截止，并保留内部软截止。

启动前父Agent报告62项机械检查通过。冻结后的分析入口增加独立输入清单核验和7个结构保存点检查，属于结果前审计实现补齐，不改变正式训练协议、数值判据或分组，不作为学习结果证据。

## Batch / Run / Time

实际批次`20260926T204123-2`。开始时间由负责启动的父Agent直接报告，不从目录名或文件修改时间反推；当时进程会话22351正在运行。当前登记尚未收到结束时间和完成记录，不填推测耗时。

## Failures / Missing Events

截至启动状态登记，尚未收到失败或完成通知；这不保证执行后续不会失败。900秒强制终止可能没有`completion.json`，届时保留真实退出码、现存文件和日志，不补造完成状态。

## Deviations

本次核对未发现实际协议与冻结协议偏离。其他实施偏离须在执行结束后根据真实记录审查，不预先宣布全程无偏离。未变更冻结设计、8775或旧结果。

## Outputs

已有来源记录为运行目录的`protocol.json`及`commits.txt`；运行日志位置为`logs/feedback-plasticity/20260926T204123-2.log`。本次不读取评价、不固定未完成目录清单、不注册Dataset、不运行Analysis。

## Record Boundary / Corrections

直接依据是父Agent真实启动消息、已读取的`commits.txt`及协议字节核对。正式开始时间不是先前工程计时、设计冻结或本次登记时间。当前只保存实施开始状态，后续结束、缺失和偏离将追加记录；不得将本次记录解释为效果验收。
