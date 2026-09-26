Status: resolved
Category: enhancement
Blocked by: 05

# 多蚁共享信息素与两段预算

用户授权多只独立模型蚂蚁参与后续基础课程；明确探索预算归零仍有返巢体力。世界只处理预算、交互和固定释放，不强制方向。基础动作保持冻结。当前阶段不涉及死亡继承。

验收：每物理步只推进一次信息素；独立世界隔离；耗尽探索预算仍可行动；实际返巢补给一次；停在巢内不反复获取返巢奖励；共享库存不超额拾取；局部观察只有接收器和自身状态。

## Answer

`ColonyEnvironment` 与78维局部观察已实现；参数和未解决的奖励投机风险见 `docs/engineering/colony-budget-interface.md`。`bash scripts/training/check.sh -q -k 'foraging_colony or local_foraging'` 的19项测试通过。尚未据此宣称模型已学会预算耗尽返巢。
