Status: resolved
Category: enhancement
Blocked by: 06

# 无食物预算耗尽返巢诊断

按 `docs/engineering/colony-return-probe.md` 固定16个无训练世界，核验预算耗尽返巢是否真实稳定，避免食物搬完提前结束掩盖失败。完成后按真实正负结果决定下一步，不增加训练预算。

## Answer

执行目录 `logs/colony-return-probe/20260926T071059-2/`；全部16个世界结束，结果和边界追加至方案文档。奖励后22/32个体发生预算返巢，信息素遮蔽后4/32；但成功者最大离巢距离仅0.920至3.122，仍需区分预算切换与普遍在巢旁徘徊。没有进一步训练。
