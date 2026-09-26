# Analysis: 生存反馈的有限配对结果

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/survival-feedback.md)
- [数据](../../data/survival-feedback/README.md)

## Question / Target Contrast

新生存反馈固定接受，相对旧反馈固定接受与全部跳过，是否减少世界内真实危险死亡或体力耗尽的失败总次数？

## Inputs and Data Freeze

批次 `20260926T161127-3`，完整散列清单与来源记录先登记，不剔除任何世界。

## Unit of Inference

两个世界种子；同世界配置、组、个体和时间相关，不按蚂蚁或窗口计算独立样本量。

## Primary Analysis

逐种子计算新固定接受减旧固定接受、新固定接受减全部跳过的失败差。两种子对两个参照都不增失败、每个参照至少一种子严格改善，才按冻结设计进入接受课程设计。允许交付下降，但不能用交付抵消失败增加。两个种子仅作有界筛选，不作显著性推断。

## Exploratory / Sensitivity Analyses

分别报告危险死亡、非危险耗尽、交付、伤害和完整四组结果，不根据结果新增阈值或系数。

## Assumptions and Diagnostics

全文件散列和源参数核对；单变量协议、2048步完整记录、外部日程同步、三个不更新对照行为一致。逐帧核对失败/复活、库存、学习反馈与提案窗口、累计奖励和写入，另核对参数更新链及记忆快照。

## Outputs

待执行正式分析。模拟日志已输出逐世界原始计数；主要判据在采样前冻结，此处登记的是完整审计和配对计算，不声称原始结果仍不可见。

## Reproduction

入口 `scripts/analyses/survival-feedback.sh`，配置 `.research/analysis/survival-feedback/A001/config.json`；结果写独立输出目录，不覆盖原始轨迹。

## Result Boundary

即使通过也只支持当前八模型库、两个新实例中的开发筛选；不能证明学习接受时机有效、记忆已训练、陌生类别分类或跨场景普适优势。

## Amendments

无研究目标或判据修订；审计读取器扩展到显式组列表及新反馈字段，不改变模拟原始轨迹。
