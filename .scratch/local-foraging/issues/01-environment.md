Status: resolved
Category: enhancement

# 隔离八接收器局部往返环境

用户已确认三类基础信号并增加未训练接收器至八个。环境不再暴露全局目标方向或距离、不接收模型释放开关，复用双信息素场并通过局部连续采样提供响应。原页面与旧检查点保留。

八项确定性检查已通过；详见 `docs/engineering/local-foraging-interface.md`。完成仅表示物理与观察接口就绪，模型输入连接冻结和真实往返学习由后续条目验证。
