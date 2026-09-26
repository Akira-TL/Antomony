# Dataset: 空载返巢奖励配对轨迹与参数

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/return-reward-ablation/README.md)
- [冻结设计](../../designs/return-reward-ablation.md)

## Dataset Identity

身份 `return-reward-ablation`，版本 `20260926T142153-2`；两个奖励配置，各10个整巢世界。采样已完成，全部原始文件由[完整散列清单](manifest.sha256)固定为分析输入。

## Research Purpose

检验取消空载返巢2分是否改善移动危险条件中的实际搬运和存活，并以不更新模型的组验证单变量修改。不作为一般自训练优势证明。

## Source and Version

来源为本项目连续模拟执行器，初始执行提交 `a8fc37b2f21cffcb7d80a5f278b304531884ec31`。两批分别记录实际提交、时间、完整协议和参数来源散列；设计固定于 `81ac692`，不依赖此前工程记录的得分筛选。

## Population and Sample Mapping

样本身份 `<variant>-moving-danger-<seed>-<arm>`。配置 `on/off` 对应空载返巢奖励2或0；种子19301、19302，组别 `learned/skip/always/mlp/rules`。每巢32只使用八份初始化各独立复制四份；推断单位为世界种子。

## Data Layers and Artifacts

原始目录 `logs/return-reward-ablation/20260926T142153-2/`，不原位修改。每批包含执行身份、世界汇总；每世界包含完整压缩轨迹、更新提案、参数与隐藏状态快照。派生结果另存分析目录。采样结束后按全部原始文件生成SHA-256清单，不筛除失败轨迹。

## Metadata / Missingness / Exclusions

动作前观察和来源对应 `tick-1`，身体及事件对应完成后的 `tick`。死亡与耗尽在累计终止计数内，真正体力耗尽为终止减危险死亡。复活后伤害累计从零重新计算，终止与复活次数不清零；尚未首次出巢的排队个体不算死亡。丢失食物归库存不算交付。没有计划排除任何世界。

## QC and Anomalies

两批执行器均完成冻结基础、动作、接受模型及源参数身份检查，20世界均达到4096步。正式分析另核对文件、事件、奖励、复活、快照和三个不更新组跨配置行为一致性。完整记录不自动表示学习有效。

## Processing and Reproduction

产生入口 `scripts/training/return-reward-ablation.sh`；两份配置在 `.research/protocols/return-reward-on.json` 与 `return-reward-off.json`。初始、每256步及终点快照并非完整世界续跑存档。后续分析只读原始文件。

## Freeze / Access / Ethics

2026-09-26 14:30:18.142436 UTC采样完成后，以完整清单及Git保存的协议固定输入。本地生成记录不强制加入Git，不外传，无真实受试者，不读取或修改原V2V。内部冻结不称外部预注册。
