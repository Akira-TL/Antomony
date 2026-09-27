# Dataset: 方向归一化配对课程原始记录

## Navigation

- [研究首页](../../../RESEARCH.md)
- [设计草案](../../../designs/plastic-projection.md)

## Dataset Identity

准备模板，尚无本批数据身份。实际实施后填写版本、来源实施对象、完整目录、清单数量和清单散列。

## Research Purpose

为两个架构、同初始化训练前后及四组配对评价提供原始记录；数据存在和机械检查不等于课程效果通过。

## Source and Version

尚无真实来源事件。冻结、代码提交、批次、真实接收时点及源文件身份待实际记录核对后填写，不使用计划值冒充事实。

## Population and Sample Mapping

计划1711/1712两个独立外层初始化，模式、训练前后、条件、回合和四组在初始化内相关。实际覆盖与缺失在实施后逐项核对，不把配对组当独立模型。

## Data Layers and Artifacts

尚无原始层文件。正式原件不原地改写；完整目录清单及自身散列入Git，大型轨迹和参数可保留本地忽略目录并登记真实位置。目录外控制台日志由实施来源单独记录。

## Metadata / Missingness / Exclusions

尚未核验。真实数据必须包含实际协议、代码身份、保存点和评价时点；任何缺失、失败或不完整都记录，不根据效果剔除或补造。

## QC and Anomalies

未执行质量核验。先核验完整清单和协议身份，再交配对内容检查；不以有文件或退出码0代替全部科学有效性。

## Processing and Reproduction

计划由`scripts/training/plastic-projection.sh`产生原始记录。实际目录不再写入后才能固定清单，首次汇总必须使用这份清单；具体生成命令和来源在实施后填写。

## Freeze / Access / Ethics

仅本地模拟，无真实生物或个人数据；原V2V只读，不修改8775。当前没有可分析数据，不登记虚构的数据对象，不把设计冻结时间当数据冻结时间。
