Status: resolved
Category: enhancement
Blocked by: 04

# 有界训练更新判断并保留留出结果

按 `docs/engineering/direction-gate-probe.md` 执行一次固定预算；所有条件独立加载相同基础参数。保留教师标签、定期判断权重、逐步接受选择和优化器状态。不依据开发结果重新挑种子或挑检查点。

完成条件：确定性接口检查通过、预算内运行完成、全部结果包括负差值保留并说明研究边界。

## 结果

固定预算已执行：1,152 条教师标注、3 个判断初始化、180 条留出轨迹；97 项训练接口回归通过。参数快照、逐步记录和全部有符号差值已保留，详见 `docs/engineering/direction-gate-results.md`。正常与短暂条件仍退化，周期条件不优于简单规则，不把本条完成等同于自训练优势成立。
