# Dataset: 固定终点的个体接受模型

## Navigation

- [研究首页](../../RESEARCH.md)
- [来源采样](../../study/memory-update-learning/README.md)
- [来源分析及负结果](../../analysis/memory-update-learning/README.md)

## Dataset Identity

身份 `memory-acceptance-models`，版本 `memory-update-learning-A001`，八个独立接受模型的全部72份间隔参数和八份损失记录。部署只使用每个体固定 `step-0200.npz`，不按后续环境选择中途参数。

## Research Purpose

将已完成分析的模型提升为可独立引用的下游数据输入，使连续部署不依赖另一分析的临时执行目录。这不是新训练、模型改进或成功证据；该来源模型只在1/4留出种子优于两参照，没有所选焦点交付改善，已经未通过原开发条件。

## Source and Version

生产分析为 `memory-update-learning` 的A001，执行提交 `7a10ca4feb9d925b7676a4f8c90e189018c8fb91`，来源结果提交 `087793b`。原目录 `.research/analysis/memory-update-learning/A001/outputs/ant-{00..07}/`，本次原样复制至 `logs/model-artifacts/memory-acceptance/ant-{00..07}/`。2026-09-26复制后逐份与来源清单核对，80文件完全一致；未改写来源目录。

## Population and Sample Mapping

目录个体00至07分别对应原基础MLP初始化种子81至88。每个体只使用自己的候选记录拟合，参数不跨蚁共享；原分析训练和留出世界不同。个体模型不是八个独立群体效果重复。

## Data Layers and Artifacts

本数据属于原分析派生参数的固定副本。80文件由 [manifest.sha256](manifest.sha256) 固定，包含初始与每25训练步至200的参数、以及原损失。原始世界和候选标签仍由来源数据集管理，不重复复制。

## Metadata / Missingness / Exclusions

未排除个体或中间快照。训练接口为140维输入、单个线性奖励差输出，共141参数；部署选择预测大于零的存活非零提案。200步终点不是本次评价后选择的最优模型。

## QC and Anomalies

来源清单81文件（含汇总）核对通过；复制的80个模型/损失文件逐份一致。下游加载再检查版本、维度、有限性及训练步数。留出失败属于模型证据边界，不作为损坏文件丢弃。

## Processing and Reproduction

只执行保留文件名和内容的目录复制，无数值变换或再次拟合。原分析入口 `scripts/analyses/memory-update-learning.sh` 与固定配置可重建训练；新数据入口是此处登记的外部目录，而非原分析执行区。

## Freeze / Access / Ethics

本机模拟参数，文件不外传；清单和来源正文进入Git。后续评价不得修改此版本。基础方向、记忆和动作模型另按其原始身份引用；没有因复制参数而增加复杂环境训练经历。
