# Analysis: 基础扰动恢复与候选后果描述

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/restoration-control.md)
- [原始数据](../../data/restoration-control/README.md)

## Question / Target Contrast

固定无危险诊断世界中，同状态恢复已知初始残差或接受当前候选，相对跳过在后续最多64步的焦点奖励及交付差。

## Inputs and Data Freeze

批次 `20260926T100314-2`，六世界70点，594文件清单与数据身份已提交 `0bee4eb`。当前已知主轨迹步数与交付，尚未计算配对参照收益；首次分析前登记计划。

## Unit of Inference

两个整巢种子；三初始化条件轨迹不同，跨条件不按候选编号配对。同巢个体和候选相关，不做总体显著性或置信区间。

## Primary Analysis

完整核验文件和模型来源散列、世界身份、快照与初始随机流重建一致、两次跳过逐字段相同、零残差恢复精确中性、分支时限及无伤害。逐世界报告候选与恢复各自接受减跳过的焦点奖励、焦点及整巢交付差：正负相同数、均值、范围。奖励容差1e-6、交付容差0。另描述入选点绝对旋转角的度数均值和最大值，不声称全轨迹角分布。

## Exploratory / Sensitivity Analyses

不追加调参、不选择个体或种子。结果只按冻结设计判断下一开发方向，不能作为部署算法改善结论。

## Assumptions and Diagnostics

恢复利用已知初始扰动，不是学到的更新，也不是最优上限。只有一个随机延续和64步短窗口，恢复可能变差。采样只覆盖非零存活候选，可能不能代表全部可改善状态。

## Outputs

待运行；计划生成逐世界结构化描述，没有训练模型或阈值。

## Reproduction

入口 `scripts/analyses/restoration-control.sh`，实现 `scripts/analyses/restoration_control.py`，配置 `.research/analysis/restoration-control/A001/config.json`。结果独占创建，不覆盖旧分析。

## Result Boundary

只用于区分课程改善机会与候选方法的不足。不能将参照答案交给部署模型，不能把变难、恢复有效或记录检查通过当成自训练优势。

## Amendments

不适用：首次结果尚未生成，没有结果后修订。
