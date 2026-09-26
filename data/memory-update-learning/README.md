# Dataset: 新基础模型的无危险接受配对数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际采样](../../study/memory-update-learning/README.md)

## Dataset Identity

身份 `memory-update-learning`，版本 `20260926T113733-2`。24个世界、301条候选记录，其中训练196条、留出105条。

## Research Purpose

用于固定预算训练每只个体的接受决策，并检验未参与拟合的世界中选择候选的短期价值。不用这批无危险短分支直接声称复杂未知环境适应有效。

## Source and Version

本项目冻结设计 `f45dfe2`，登记提交及实际执行提交 `e78a3f7`。原始目录 `logs/memory-update-learning/20260926T113733-2/`，来源检查点、配置、分区及散列由 `execution.json` 记录。

## Population and Sample Mapping

`initial-0`为零残差，`initial-1`为同向第四接收器范数0.75扰动。训练种子16101至16108，留出16111至16114。目录、时点及焦点个体组成候选身份；整巢种子是推断单位，两条件在种子内配对。

## Data Layers and Artifacts

原始层2738文件，包括1344组参数及残差双文件，全部由 [manifest.sha256](manifest.sha256) 固定。没有原地改写、样本排除或标签变换；后续拟合模型和估计另存分析目录。

## Metadata / Missingness / Exclusions

只收非零存活候选，不依据后续收益选择。零候选及终止点计数保留；每世界最多32条，不足不追加。所有301条保留，不能借用其他个体或留出记录弥补某个体训练样本不足。

## QC and Anomalies

采样正常退出，来源检查点和主轨迹残差不变，所有世界无死亡。清单覆盖整个运行目录。分析前仍须核对完整清单、帧数、快照、分区及配对字段；此记录不替代该核验。

## Processing and Reproduction

运行入口 `scripts/training/memory-update-learning.sh`，核心为 `src/mathhackson/training/foraging/update_curriculum.py` 和 `candidate_probe.py`。清单为原始文件的SHA-256，没有效果变换。冻结后分析只读该目录，不重采样替换。

## Freeze / Access / Ethics

本记录与清单提交后作为固定分析输入。原始轨迹与模型保留本地忽略目录，通过清单追踪，不强制进入Git。没有真实生物、个人资料或外传操作。尚未开始接受模型拟合及留出计算。
