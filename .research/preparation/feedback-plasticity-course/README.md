# 可塑方向基础课程的实施与分析准备

这是结果前准备，不是Study已经开始、Dataset已产生或Analysis已经运行。当前正式实验没有启动，本目录不提供任何学习效果证据。

## 固定入口

- 科研设计：`designs/feedback-plasticity-course.md`，Design 22。
- 唯一训练协议：`.research/protocols/feedback-plasticity-course.json`。
- 启动入口：`scripts/training/feedback-plasticity.sh --plan .research/protocols/feedback-plasticity-course.json --freeze-commit <数据库登记的真实提交>`。
- 结果前分析计划：本目录`analysis-plan.md`。
- 研究状态通过`research-db`正式CLI登记，不直接修改SQLite。

## 真实启动时

先记录真实UTC启动时间和运行目录，再将`study-template.json`中的空`started_at`替换为该时间，将实施正文准备到`study/feedback-plasticity-course/README.md`并遵守固定Study章节；记录`in_progress`，不是`completed`。不能把本目录准备时间或此前工程计时的时间当作正式实施开始，也不能提前登记已完成样本/测量。

Study正文应明确模型初始化1701/1702对应的源文件、3种条件与4种写入干预、实际执行提交与冻结提交、900秒截止、输出位置以及未完成情况。运行返回前不填结束时间；外层强制终止时即使没有`completion.json`，也保留退出码、完整现存文件和日志。不为完成流程而重训或补齐缺失实验。

## 数据产生后且首次汇总前

1. 固定整个实际输出目录的文件清单、大小和SHA-256，包括未完成、负值和日志；登记本课程Dataset。数据身份采用实际批次，不覆盖工程计时目录。
2. 将本目录`analysis-plan.md`逐字保留为结果前版本来源，再建立`analysis/feedback-plasticity-course/README.md`；填写实际数据身份和输入链接，不改主要对比、窗口或判据。
3. 填写`analysis-template.json`，关联真实已登记Dataset，并登记`planned`。CLI要求存在Dataset，不得用旧候选数据代替或凭空登记未产生的原始轨迹。
4. 独立提交分析输入、配置及计划，按Analysis Attempt契约在`.research/analysis/feedback-plasticity-course/A001/`准备实际路径配置，绑定真实Git提交，然后才运行分析入口。
5. 输出与Observation依据真实结果登记；两个初始化全部完成也不等于通过，全部门槛必须如实逐项报告。

## 模板边界

两个JSON均是未满足执行条件的模板，含空必需字段，不能原样送给CLI冒充已实施状态。正式Study、Dataset和Analysis目前没有编号。设计、来源检查及完整预分析方案由本轮Git提交固定；随后数据库可回填该提交并推进设计状态。
