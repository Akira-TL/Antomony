# Research

## Objective

检验蚁群模型能否学会控制自己的参数更新，并在基础能力相当时，相比普通多层感知机（Multilayer Perceptron, MLP）和无神经网络规则，更有效地适应未见环境。重点是可行性及适应能力，不是宣称所有组成机制首次出现。

## Applicable Standards

研究对象为本地模拟个体，不涉及真实受试者。自愿参考NeurIPS论文检查清单的声明范围、复现、随机性及成本报告要求；它不是赛事强制规则或研究方法。2026-09-26核验来源见[总体设计](designs/ant-self-training-adaptation.md)。内部Git冻结不等于外部预注册。

## Current Loop

INTERPRETATION

## Active Uncertainty

连续更新已经在本批次表现出相对不更新的收益，但学习接受没有优于固定接受。尚未解决的是：如何让接受决策在未见条件下识别有用和有害的更新，而不是仅减少更新次数？新课程的收益必须在新的评价数据中验证，不能用现有失败批次调参。

## Current State

纯方向动作底座、八接收器、固定信息素释放、真实拾取交付、多蚁共享场、独立模型、探索预算及返巢体力已实现。有限巢穴源半径1.5，远处仍依靠局部轨迹；不向策略提供目标距离、食物或巢穴全局坐标。

[完整连续对照](analysis/continuous-adaptation/README.md)已完成80世界并逐帧重建。三复杂条件等权后，学习接受比全部跳过平均多交付5.25份，却比固定接受少1份；危险条件比固定接受平均多死亡1只。仅2/4种子、1/3条件同时优于两参照，未通过采用条件。282次真实写入说明更新确实发生，不说明接受时机学习有效。

正常条件三更新组逐帧一致；所有五组均完成48份交付，但不能据此宣称基础能力统计等效。普通MLP总1732参数，自训练候选1697参数；候选新增64记忆连接为零且冻结，本轮不是训练后循环记忆的效益检验。主动追逐和死亡继承未实现；移动危险仅沿预定路线移动。

## Active Work

优先把已完成对照的真实轨迹、固定间隔参数和接受/跳过记录接入只读验收控制页面，让用户能检查搬运、死亡、写入时刻和参数变化。复用现有渲染能力，不重建宣传页面，不把旧8771/8772模型显示成此次研究模型。原始数据、失败条件和负差值全部可见。

科学上停止当前接受课程及本批次调参。现有证据不能把问题归为“完全没有可用更新”；固定接受已经更好。若继续探索，下一条路线应专门改变接受决策的训练信号或可泛化输入，并另定有限预算、新评价种子与停止条件，而不是重复扩大参数或训练轮数。该新路线尚未冻结、未执行。

## Open Threads

- [近似规模MLP资格](analysis/matched-foundation/README.md)：三组均交付128/128并通过最低门槛，返回及探索范围仍不同。采用固定2400轮终点，不延长训练。
- [保留基础后的接受课程](analysis/memory-update-learning/README.md)：无危险训练及留出完成，仅1/4种子通过、没有焦点交付改善。当前连续对照没有推翻该失败。
- [无线索探索补课](analysis/exploration-course/README.md)：半径略增但返回和耗尽恶化，停止采用；[基础资格](analysis/foundation-qualification/README.md)及[初始接受学习](analysis/basic-update-learning/README.md)的失败均保留。
- [候选价值诊断](analysis/trust-candidate-value/README.md)、[恢复参照](analysis/restoration-control/README.md)和[窗口比较](analysis/feedback-window/README.md)只提供开发线索，不是连续策略收益证据。已知恢复答案不能部署，64步窗口不优于16步。
- 完整参数/世界/随机状态无缝恢复尚未实现；当前固定间隔文件是参数快照，实际行为另由完整轨迹保存。
- 旧连续预测分支暂停，未冻结、未执行；旧状态保存在 `ceea87c`，数据库迁移为 `a382b08`。原候选描述分析的首次成功登记晚于结果这一偏离仍有效，见[原分析](analysis/candidate-update-value/README.md)。

## Key Decisions

每只模型的参数、记忆、随机流及更新缓冲独立；同组共享信息素，组间场、库存和个体隔离，外部干预配对同步。规则组完全不使用神经网络。

流程为推理、实际后果、更新提案、接受或跳过、下一次推理。当前在线只修改45个陌生信号方向参数，动作、基础、记忆及接受模型均冻结。候选根据已发生窗口反馈，不使用未经校准的价值输出或未来分支。

完整连续对照失败的是当前学习接受方案的采用条件，不等于所有自训练不可行。保留全部负结果，不预设模型获胜，不在本批次挑选阈值或中途模型。原V2V只读，不改网络、代理、nginx或ForgeRelay。

## Navigation

- [项目入口](README.md)
- [总体对照设计](designs/ant-self-training-adaptation.md)
- [研究结构](research-tree/README.md)
- [连续对照结果](analysis/continuous-adaptation/README.md)
- [连续原始数据](data/continuous-adaptation/README.md)
- [固定接受模型](data/memory-acceptance-models/README.md)
- [全部分析](analysis/README.md)

## References

结构化状态：`.research/research.sqlite`。连续批次 `logs/continuous-adaptation/20260926T123655-2/`，完整来源、清单、实际实施及分析代码由关联对象固定。原始模型与大体积轨迹不强制加入Git，不能以代码提交代替参数身份。
