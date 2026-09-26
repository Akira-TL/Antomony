# Dataset: 行动概率约束候选配对数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/trust-candidate-value/README.md)

## Dataset Identity

身份 `trust-candidate-value`，版本 `20260926T091027-2`，本项目模拟原始数据。四条主轨迹，58对分支。

## Research Purpose

检查一次概率约束候选相对跳过是否产生实际后果差异；不用于宣称更新决策已学会或未见环境优势。

## Source and Version

来自已登记冻结设计及实际采样，执行提交 `51cca61`。原始目录 `logs/trust-candidate-value/20260926T091027-2/`；输入模型、协议和实际参数散列由 `execution.json` 记录。

## Population and Sample Mapping

四个条件与种子身份和实施记录一致，点数为16、16、13、13。候选以条件、种子、物理步、焦点个体识别；两个整巢世界种子才是独立单位，同巢候选不能当58个独立重复。

## Data Layers and Artifacts

原始层包含执行元数据、世界记录、四份配对行记录、四份压缩主轨迹和168组模型及残差双文件。`manifest.sha256` 固定全部346个文件，参数快照用于追溯而非训练效果声明。暂无整理层或派生结果，不原地改写输入。

## Metadata / Missingness / Exclusions

主轨迹全部跳过，分支之后均冻结；零增量和终止点按结果前规则不纳入存活分支，计数保留。没有为了凑满上限追加或填充样本。所有点均保留，不做结果后排除。

## QC and Anomalies

采集进程通过来源检查与主轨迹无参数写入检查。分析前另校验清单散列、字段、配对唯一性、计数、时限、概率约束和快照残差零值；当前不以文件存在代替这些检查已完成。

## Processing and Reproduction

生成入口 `scripts/training/trust-candidate-value.sh`；没有数据变换。清单由原始文件逐一计算SHA-256，不重采样生成所谓相同原件。后续分析只读该快照。

## Freeze / Access / Ethics

清单与本记录提交后作为分析输入快照。本地原始模型文件不全部进入Git；其位置和散列可追溯。数据不涉及真实生物或个人资料，未向外部服务上传。
