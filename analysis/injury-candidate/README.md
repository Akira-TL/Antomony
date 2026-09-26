# Analysis: 伤害候选与接受判断

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/injury-candidate.md)
- [输入数据](../../data/injury-candidate/README.md)

## Question / Target Contrast

既有两个学习接受主轨迹中，含实际伤害的非零更新候选在相同状态下接受一次相对跳过，后续最多128步焦点个体的奖励、实际交付、危险死亡、体力耗尽及伤害差，以及原接受判断与这些后果的对应关系。

## Inputs and Data Freeze

输入为 `injury-candidate` 版本 `20260926T145121-2`，完整清单冻结。原始采样点和运行输出可见，不称盲法；分析规则登记在正式计算之前。

## Unit of Inference

两个世界种子，同世界点相关，每点只有一次共同随机流延续。不对32点作独立样本推断，不把分支差之和当整巢收益。

## Primary Analysis

先核对源及本批散列、父参数记忆、分支连续帧、食物守恒和累计事件。差值为接受减跳过。按焦点奖励差大于1e-6、小于-1e-6或其余分为改善、恶化、持平，与原接受/拒绝交叉逐世界报告；另保留全部交付、死亡、耗尽、伤害及整巢差。

## Exploratory / Sensitivity Analyses

仅描述原拒绝中少死亡及多交付的点，不追加阈值筛选、窗口搜索或训练。两个世界均存在改善却被拒绝的点时，优先进一步设计接受输入/课程；否则不能单独归罪接受模型。

## Assumptions and Diagnostics

分支只改变焦点一次参数，其余未来更新冻结，不能外推为持续接受该候选族的收益。死亡后伤害清零须按复活事件累计，不把回巢身体位置当奖励；提前终止必须符合世界时限或食物交付完成。

## Outputs

尚未执行正式分析，不预判结果。全部行和逐世界分类将存于 `summary.json`。

## Reproduction

`bash scripts/analyses/injury-candidate.sh`，配置 `.research/analysis/injury-candidate/A001/config.json`；拒绝覆盖已有结果。

## Result Boundary

开发诊断，不是新的适应性留出验证。未来分支不回写在线模型，不用危险分支标签重训。奖励改善不自动等同少死亡或交付提高。

## Amendments

初始分析，无结果后修订。
