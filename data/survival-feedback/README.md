# Dataset: 生存反馈连续轨迹与参数

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/survival-feedback/README.md)

## Dataset Identity

`survival-feedback`，版本 `20260926T161127-3`，16世界原始连续轨迹。

## Research Purpose

评价生存反馈是否让固定接受减少失败，不通过搬运总分替代生存，也不使用旧接受模型代表新目标。

## Source and Version

本机模拟，实际代码 `0bc891a`，收到时间2026-09-26T16:12:31.876192+00:00。执行记录包含原始参数文件身份和协议散列。

## Population and Sample Mapping

19501、19502两个世界种子；每个种子含两配置和四组。单位是世界，组内8只、2048步、参数快照均为相关测量。

## Data Layers and Artifacts

原始数据位于 `logs/survival-feedback/20260926T161127-3/`；[完整散列清单](manifest.sha256)固定所有文件。派生分析另存，不覆盖原始数据。

## Metadata / Missingness / Exclusions

无剔除，16个世界均保存结果。生存配置额外记录学习反馈与新增伤害，旧配置这两字段为空属于预定记录方式，不按缺失反馈填零。

## QC and Anomalies

文件完整性、逐帧事件、实际反馈、参数写入链及三个无更新组的行为一致性在正式分析入口强制核验；未通过则停止解释。不对未通过的结果静默修补。

## Processing and Reproduction

采样入口 `scripts/training/survival-feedback.sh`；散列由全部原始文件生成，读取配置和执行记录不改变数据。分析使用独立目录及登记版本。

## Freeze / Access / Ethics

清单及人类来源在首次正式分析前提交登记；原始大文件只本机保存，不要求提交Git。全为模拟，无受试者。无网络、V2V或8774修改。
