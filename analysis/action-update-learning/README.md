# Analysis: 四步候选的新旧输入接受学习对照

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/action-update-learning.md)
- [原始数据](../../data/action-update-learning/README.md)

## Question / Target Contrast

在相同无危险基础候选上，新增动作分布输入能否比旧参数输入更可靠地接受有益更新？主要比较两留出种子上的选择收益差，同时对照全跳过与全接受。

## Inputs and Data Freeze

固定批次 `20260926T173720-2`，数据提交 `662b601`，1306文件、12世界、39配对。已知采样数量为训练25、留出14，原始父轨迹计数已见；尚未拟合或查看两输入配置的留出预测收益，不将本计划称作结果全盲。

## Unit of Inference

整巢世界种子。四训练、两留出；同种子两初始条件、个体与配对候选相关。不得把39配对当39次独立实验。

## Primary Analysis

先审计完整清单、协议、模型身份、4502帧、640组快照和配对约束。每个体分别用原140维与新增178维输入拟合200步，学习率与权重衰减均0.01，每25步保存，只有最后一步进入评价。两配置全部拟合完才读取留出标签进行预测。

预测大于0则接受；每条真实奖励差乘接受指示，按世界求均值，再按初始条件等权。分别报告两个种子，不计算显著性。新增输入须两种子均严格优于0和全接受、不差于旧输入且至少一处更好，并有真实焦点交付改善，才继续设计生存迁移。字段 `development_continue` 仅为这一基础课程门槛，不是部署资格。

## Exploratory / Sensitivity Analyses

描述各个体正负训练标签数量、最终训练损失、留出预测均方误差和交付变化。无样本个体保留零更新标记，不借用其他个体数据。没有额外学习率或阈值搜索。

## Assumptions and Diagnostics

新增特征必须在未来分支前取得、版本与维数正确；未来后果只能为标签。两个输入配置必须对应相同候选数和全接受收益。额外38参数与动作信息共同改变，不能仅归因于信息。样本非常少，可能无法学习或泛化；不能据此推广为自训练原理无效。

## Outputs

待执行：两套八个体检查点、训练损失、逐世界汇总及继续判定。输出目录 `.research/analysis/action-update-learning/A001/outputs/`。

## Reproduction

入口 `scripts/analyses/action-update-learning.sh`，配置 `.research/analysis/action-update-learning/A001/config.json`，实现 `action_acceptance.py` 及已有课程/审计模块。单线程，120秒上限；输出目录拒绝覆盖，不重采数据。

## Result Boundary

待结果。无危险任务奖励差不等于生存改善，独立分支的选择收益不能拼接成连续策略收益；本批不解除生存反馈模型限制，不更新8774。

## Amendments

不适用：没有结果前或结果后修订。
