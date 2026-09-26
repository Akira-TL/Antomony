# Data

## Objects

- [首步方向数据](first-action-outcomes/README.md) - 209文件、8状态128分支，完整清单固定。

- [后到反馈归因数据](credit-history/README.md) - 四世界、54个共同状态，完整清单固定。

- [四步动作输入配对](action-update-learning/README.md) - 12世界39配对，训练25、留出14。

- [四步与十六步窗口](window-cadence/README.md) - 16世界轨迹与参数，完整清单固定。

- [负反馈提前判断](feedback-trigger/README.md) - 16世界原始轨迹与参数，完整清单固定。

- [生存反馈轨迹](survival-feedback/README.md) - 16世界原始轨迹及参数，完整清单冻结。

- [历史基线连续数据](historical-baseline/README.md) - 20世界，原始轨迹与基线快照已冻结。

- [伤害候选轨迹](injury-candidate/README.md) - 32对、64分支，完整清单固定。

- [返巢奖励配对数据](return-reward-ablation/README.md) - 已完成20世界并固定清单，完整性与配对核对通过。

- [连续复杂来源数据](continuous-adaptation/README.md) - 80世界、6634文件，未按结果筛选。

- [固定个体接受模型](memory-acceptance-models/README.md) - 原分析参数的完整固定副本，来源课程未通过采用条件。

- [近似规模模型基础数据](matched-foundation/README.md) - 253文件，含104参数快照与48世界轨迹。

- [新基础接受配对数据](memory-update-learning/README.md) - 301条记录、2738文件和1344组快照。

- [探索课程数据](exploration-course/README.md) - 245文件，含72组参数快照。

- [三组基础轨迹](foundation-qualification/README.md) - 48世界、52份原始文件。

- [反馈窗口配对数据](feedback-window/README.md) - 28条记录、588份文件。

- [基础扰动恢复参照](restoration-control/README.md) - 70点、594份原始文件。

- [无危险基础更新数据](basic-update-learning/README.md) - 225对、1426份原始文件。

- [行动概率约束候选数据](trust-candidate-value/README.md) - 58对，完整346文件清单。
- [候选更新短期价值](candidate-update-value/README.md) - 本地模拟原始配对数据，72 点。

## Relations

- [首步方向数据](first-action-outcomes/README.md) → [实施](../study/first-action-outcomes/README.md)、[分析](../analysis/first-action-outcomes/README.md)。
- [窗口父轨迹](window-cadence/README.md) → [首步方向诊断](../analysis/first-action-outcomes/README.md)。

- [窗口原始数据](window-cadence/README.md) → [写入约束诊断](../analysis/update-constraints/README.md)。

- [后到归因数据](credit-history/README.md) → [实施](../study/credit-history/README.md)、[分析](../analysis/credit-history/README.md)。

- [动作输入数据](action-update-learning/README.md) → [实际采样](../study/action-update-learning/README.md)、[分析](../analysis/action-update-learning/README.md)。

- [窗口数据](window-cadence/README.md) → [实施](../study/window-cadence/README.md)、[分析](../analysis/window-cadence/README.md)、[动作诊断](../analysis/candidate-steering/README.md)。

- [负反馈提前判断](feedback-trigger/README.md) → [实施](../study/feedback-trigger/README.md)、[分析](../analysis/feedback-trigger/README.md)。

- [生存反馈](survival-feedback/README.md) → [实施](../study/survival-feedback/README.md)、[配对分析](../analysis/survival-feedback/README.md)、[时序诊断](../analysis/survival-timing/README.md)。

- [历史基线数据](historical-baseline/README.md) → [实施](../study/historical-baseline/README.md)、[分析](../analysis/historical-baseline/README.md)。

- [伤害候选轨迹](injury-candidate/README.md) → [实际实施](../study/injury-candidate/README.md)、[分析](../analysis/injury-candidate/README.md)。

- [返巢奖励配对数据](return-reward-ablation/README.md) → [实际实施](../study/return-reward-ablation/README.md)、[配对分析](../analysis/return-reward-ablation/README.md)。

- [连续数据](continuous-adaptation/README.md) → [实际实施](../study/continuous-adaptation/README.md)、[固定接受来源](memory-acceptance-models/README.md)、[MLP来源](matched-foundation/README.md)、[连续分析](../analysis/continuous-adaptation/README.md)。

- [固定接受模型](memory-acceptance-models/README.md) → [来源分析](../analysis/memory-update-learning/README.md)、[来源采样](../study/memory-update-learning/README.md)。

- [近似规模基础数据](matched-foundation/README.md) → [实际实施](../study/matched-foundation/README.md)、[资格分析](../analysis/matched-foundation/README.md)。

- [新基础接受数据](memory-update-learning/README.md) → [实际采样](../study/memory-update-learning/README.md)、[接受学习结果](../analysis/memory-update-learning/README.md)。

- [探索课程数据](exploration-course/README.md) → [实际实施](../study/exploration-course/README.md)、[配对评价](../analysis/exploration-course/README.md)。

- [基础轨迹](foundation-qualification/README.md) → [实际采样](../study/foundation-qualification/README.md)、[描述分析](../analysis/foundation-qualification/README.md)。

- [反馈窗口数据](feedback-window/README.md) → [实际采样](../study/feedback-window/README.md)、[配对分析](../analysis/feedback-window/README.md)。

- [恢复参照数据](restoration-control/README.md) → [实际采样](../study/restoration-control/README.md)、[后果分析](../analysis/restoration-control/README.md)。

- [基础更新数据](basic-update-learning/README.md) → [实际采样](../study/basic-update-learning/README.md)、[训练与留出分析](../analysis/basic-update-learning/README.md)。

- [概率约束数据](trust-candidate-value/README.md) → [实际采样](../study/trust-candidate-value/README.md)、[描述分析](../analysis/trust-candidate-value/README.md)。
- [本数据](candidate-update-value/README.md) → [实际采样](../study/candidate-update-value/README.md)、[描述分析](../analysis/candidate-update-value/README.md)。
