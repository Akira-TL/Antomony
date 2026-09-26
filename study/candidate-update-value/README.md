# Study: 候选更新短期价值采样

## Navigation

- [Research](../../RESEARCH.md)
- [Design](../../designs/candidate-update-value.md)

## Study Identity

`candidate-update-value`，计算模拟采样，已完成。2026-09-26 执行一次完整批次，不含冒烟运行。

## Source and Experimental Units

9501、9502 两个世界种子，分别运行无害和持续伤害条件。每条轨迹八只独立模型；条件间配对，同轨迹候选不是独立样本。

## Actual Groups / Exposure / Intervention

主轨迹始终跳过修改。每个选中候选分出接受和跳过两个世界，只有焦点个体接受一次，之后全部停止学习；分支不回写主轨迹。

## Sample Collection and Processing

实际选中无害 9501 的 31 点、无害 9502 的 24 点、持续伤害 9501 的 5 点和持续伤害 9502 的 12 点，共 72 点。按冻结规则排除零增量和终止候选，不补满 128 上限，不按分支结果筛选。

## Assays and Measurements

72 对短期分支均返回，实际时长不超过 64 步。输出当前局部观察、候选参数和配对后续反馈、交付、伤害等；尚未作结果分布分析。

## Protocol / Materials / Instruments

设计冻结提交 `74a777a`；执行提交 `790232d`；参数配置 `.research/protocols/candidate-update-value.json`。单线程 PyTorch，项目既有环境和锁文件。九份输入模型哈希保存于执行记录，结束时源文件哈希复核通过。

## Batch / Run / Time

批次 `20260926T081311-2`，脚本记录的 UTC 启动时间 08:13:11；会话于 08:14:22 前已确认进程退出码为 0。后者是完成确认上界，不冒充精确退出时间。命令为 `bash scripts/training/candidate-value.sh`。

## Failures / Missing Events

无程序失败。自然提前终止和零增量使候选少于上限，属于预定设计行为，不是缺失失败。完整批次之外的 9599 冒烟不进入分析。

## Deviations

无已知设计偏离；未追加种子、代数、时限或参数变体。

## Outputs

原始目录 `logs/candidate-value/20260926T081311-2/`，含 `execution.json`、`worlds.jsonl`、四组 `pairs.jsonl`、主轨迹压缩记录及每 64 步/结束的参数快照。运行日志 `logs/candidate-value-20260926T081311-2.log`。交接至[数据集](../../data/candidate-update-value/README.md)。

## Record Boundary / Corrections

以上为命令输出和执行记录的事后事实整理，不是结果有效性判断。没有修改冻结设计或既有旧草案。分支是模拟评价，不是模型实际经历的未来。
