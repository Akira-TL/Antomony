Status: resolved
Category: enhancement
Blocked by: 01

# 建立有预算、快照与闭环验收的动作课程

## 范围与完成依据

`scripts/training/direction-motor.sh` 调用纯方向训练，不启动旧服务。保存训练初始、固定间隔和最终权重以及完整验收轨迹；对所有指定种子逐一验收，不挑选成功种子。参数、边界和运行前的工程门槛见 `docs/engineering/direction-motor-training.md`。

小步真实训练测试覆盖快照间隔、实际参数变化与冻结恢复。本条关闭仅表示训练与检查管线具备，不表示正式预算训练通过，也不表示自修改有收益。真实运行结果另记录，外部预训练优化器能否复用另行核验。
