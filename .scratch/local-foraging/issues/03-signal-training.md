Status: resolved
Category: enhancement
Blocked by: 02

# 执行基础局部信号预训练及真实轨迹诊断

固定 1200 次监督更新，预留第 4 至 8 接收器连接始终冻结。保存每 100 次权重，在 12 个独立生成场景执行完整感知与双信息素遮蔽的 24 条轨迹。运行配置见 `docs/engineering/local-signal-curriculum.md`。

成功或失败均如实保留，合成方向误差不替代实际交付；失败后根据真实轨迹决定奖励课程和环境可辨别性校准，不补全局目标方向。

## 结果

课程与24条诊断轨迹均已执行；完整感知0拾取0交付，合成平均方向误差48.89度。参见 `docs/engineering/local-signal-results.md`。条目完成仅表示固定诊断已结束，基础信号和往返能力未通过。
