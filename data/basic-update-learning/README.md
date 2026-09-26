# Dataset: 无危险基础更新配对数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/basic-update-learning/README.md)

## Dataset Identity

身份 `basic-update-learning`，版本 `20260926T094431-2`。16条主轨迹、225对候选，训练170对、留出55对。

## Research Purpose

用于拟合每只个体的接受决策并在固定留出种子上描述选择价值，不用于宣称复杂环境适应已成立。

## Source and Version

来自本项目冻结课程，执行提交 `e927445`。原始目录 `logs/basic-update-learning/20260926T094431-2/`，实际配置、分区和来源散列均在 `execution.json`。

## Population and Sample Mapping

`initial-0`为零残差，`initial-1`为范数0.75；目录继续含种子编号，候选由目录、步数、焦点个体标识。11101至11106训练，11111/11112留出。两初始化条件配对；整巢种子为推断单位。

## Data Layers and Artifacts

原始层1426文件，全部由 [manifest.sha256](manifest.sha256) 固定散列，含696组参数及残差双文件。没有改写原始层或进行样本排除；训练模型将另存分析执行目录。

## Metadata / Missingness / Exclusions

主轨迹全部跳过；分支接受一次后冻结。零候选和终止点按结果前规则计数不入选，不增加世界补满上限。所有实际225对保留；个体样本不足须显式报告，不能借用其他个体或留出样本。

## QC and Anomalies

采样通过来源不变和主轨迹残差不变检查，所有世界零死亡。分析前另核验完整清单、快照、身份与配对字段，不以此记录代替完整分析校验。

## Processing and Reproduction

入口 `scripts/training/basic-update-learning.sh`；没有结果变换。SHA-256清单由原始文件计算，未来分析只读该批文件，不重采样替换。

## Freeze / Access / Ethics

清单及本记录提交后固定分析输入。大文件保留本地，不强制纳入Git；无真实生物、个人资料或外传操作。
