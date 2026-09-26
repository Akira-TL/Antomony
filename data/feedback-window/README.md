# Dataset: 反馈时间范围配对数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/feedback-window/README.md)

## Dataset Identity

身份 `feedback-window`，版本 `20260926T101927-2`，28条候选记录，预期构成14个共同状态的窗口配对。

## Research Purpose

同状态比较64步与16步反馈候选，不改变未来评价范围，不训练接受模型，不作最终复杂环境优势结论。

## Source and Version

执行提交 `1d59700`，目录 `logs/feedback-window/20260926T101927-2/`。每组执行元数据保存完整配置、协议及模型散列。

## Population and Sample Mapping

两个整巢种子13101/13102，窗口16/64，候选按种子、时间点、焦点个体配对。独立单位是整巢种子而非28条候选。

## Data Layers and Artifacts

588份原始文件全部由 [manifest.sha256](manifest.sha256) 固定，含288组模型与残差快照。原始层未修改，分析输出另存。

## Metadata / Missingness / Exclusions

包含零候选，不依据不同窗口的非零集合重新选人。终止状态不入选，实际每世界7点；没有补样或结果后排除。

## QC and Anomalies

采样检查源模型、主轨迹残差未变，单点重复跳过相同。跨窗口完整主轨迹、观察、记忆、参照及身份匹配由后续分析执行，失败则停止解释。

## Processing and Reproduction

入口 `scripts/training/feedback-window.sh`；无数据变换，逐文件计算SHA-256。分析只读取这批快照，不覆盖重采样。

## Freeze / Access / Ethics

清单及身份提交后作为分析输入，大型文件本地保存且有来源追踪。纯模拟，无真实生物、个人数据或外传操作。
