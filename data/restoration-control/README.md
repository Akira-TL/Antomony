# Dataset: 基础扰动恢复参照数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/restoration-control/README.md)

## Dataset Identity

身份 `restoration-control`，版本 `20260926T100314-2`，六世界70点，每点包含两种干预各自的配对延续。

## Research Purpose

区分基础扰动缺少可修正偏差与当前候选未找到改善，不拟合接受模型，不提供最终复杂环境比较结论。

## Source and Version

执行提交 `a4c7b10`，原始目录 `logs/restoration-control/20260926T100314-2/`；三条件元数据各自固定协议、模型路径及散列。

## Population and Sample Mapping

`zero`、`random`、`coherent`分别为零、随机、第四接收器同向初始化，均有12101/12102两个世界。身份由条件、种子、步数、焦点个体共同确定。整巢种子为推断单位，候选相关。

## Data Layers and Artifacts

原始层594份文件由 [manifest.sha256](manifest.sha256) 固定，含288组模型和残差双文件。没有修改原始层；分析派生结果另存。

## Metadata / Missingness / Exclusions

按冻结资格采集70点，没有为补满72追加；零和终止候选计数保留，不按恢复好坏删选。两次跳过属于机械重复检查，不作为额外独立样本。

## QC and Anomalies

采样通过源模型未变、主轨迹残差未变和重复跳过一致检查。分析另检查完整清单、快照可重建性、零恢复中性、身份与角度，不以文件存在替代完整校验。

## Processing and Reproduction

入口 `scripts/training/restoration-control.sh`。原始文件逐一计算SHA-256，无数据筛选或变换；只读本批快照，不覆盖重采样。

## Freeze / Access / Ethics

清单提交后固定输入，大型参数保留本地并以散列追溯。仅计算模拟，无真实生物、个人资料或外传。
