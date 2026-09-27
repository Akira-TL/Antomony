# 方向归一化配对研究登记准备

当前只有设计草案及登记模板，不是已运行记录。以`designs/plastic-projection.md`为科学规则；本目录不能代替设计冻结或真实开始事件。

- `study-start-template.json`与`study-record-template.md`：收到实际运行目录、UTC开始时刻、真实代码及冻结提交后，立即填入规范实施记录并登记`in_progress`，不要等结束再追溯开始。两个模式顺序运行，以真实启动状态记录，不能预填全部样本完成。
- `study-completion-template.json`：退出后填真实状态、实际样本、退出码、完成记录及其时间来源。失败使用`aborted`，不以文件存在伪造完整完成；保留超时与缺失。
- `dataset-template.json`与`dataset-record-template.md`：数据不再写入后，固定完整输入清单及自身散列，再核验和登记。版本、接收时间、来源和样本必须从实际实施填写，不能使用假目录。
- `analysis-template.json`与`analysis-plan.md`：作为结果前准备；数据身份登记后、首次汇总前，生成正式正文与独立A001配置，登记`planned`。只有结果真正产生后才能记录执行尝试完成和观察。

模板中的空值与`待填写`均是有意的未就绪标记，禁止原样传给登记工具。实际人类正文分别落到`study/plastic-projection/README.md`、`data/plastic-projection/README.md`、`analysis/plastic-projection/README.md`，并改正导航层级；这些路径当前只是准备目标，不代表对象存在。

计划启动形式为`timeout --signal=KILL 900s bash scripts/training/plastic-projection.sh <实际冻结提交>`。整批900秒硬限，脚本内旧模式与候选模式各440秒外部强限，各协议420秒内部软限。真实退出前不填写完成时刻，文件修改时间不冒充进程精确退出时间，登记时刻也不冒充精确启动时刻。
