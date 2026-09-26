Status: resolved
Category: enhancement
Blocked by: 48, 49

# 隔离生存学习反馈与旧奖励

用户已明确生存优先。新增默认关闭的反馈配置，保留环境和历史奖励，不覆盖旧接受模型或8774。

## 验收

- 生存反馈只依赖当步自身伤害和终止，死亡和耗尽不重复计数。
- 探索、拾取、交付、空载补给和复活不会产生正学习反馈。
- 复活清空身体伤害不产生负伤害差或正奖励，等待复活不反复惩罚。
- 新配置不能使用旧学习接受组；旧配置默认行为不变。
- 逐帧保存环境奖励、真正学习反馈及当步伤害，验证反馈进入提案窗口。
- 本任务只是工程接口与确定性测试，不声称已训练成功。新的效果试验须另冻结设计。

## 结果

`test_survival_feedback.py`、`test_continuous.py`、`test_revival.py`、`test_return_reward_audit.py`、`test_baseline_audit.py`、`test_return_reward_protocol.py` 共48项通过。真实执行器的连续死亡测试证实原始奖励约负3而提案收到负2，6步共12次死亡不重复计数，复活清零不产生正反馈。旧协议、纯规则隔离及并行重建测试通过。[接口说明](../../../docs/engineering/survival-feedback.md)。
