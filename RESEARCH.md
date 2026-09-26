# Research

## Objective

检验蚁群中的模型能否学会控制自己的参数更新，并在基础能力相当时，相比普通多层感知机（Multilayer Perceptron, MLP）和无神经网络的规则算法，更有效地适应未见环境。重点是可行性和适应能力，不是宣称所有组成机制首次出现。

## Applicable Standards

研究对象为模拟个体，不涉及真实动物或人类受试者。自愿参考 NeurIPS 论文检查清单中的声明范围、复现、调参、随机性及成本报告要求；它不是本赛事的强制规则，也不代替研究设计。2026-09-26 核验的来源见[对照设计](designs/ant-self-training-adaptation.md)。内部 Git 冻结不等于外部预注册。

## Current Loop

DESIGN

## Active Uncertainty

只看到局部多维信号、自身状态和已经发生的反馈时，模型能否学会接受有用更新、跳过有害更新，并在未见信号组合和动力学中改善搬运与存活？必须区分候选更新本身无效、决策学不会、基础能力不匹配三种失败原因。

## Current State

纯方向动作底座已实现并可冻结；信息素往返、多蚁共享场、独立模型、探索预算与返巢储备已实现。动作仅接收身体坐标方向，不接收距离或全局巢穴位置。八接收器中前三个用于基础来源，后五个保留给陌生响应；信息素由代码释放。

目前有开发试跑和工程检查，没有完成本问题的确认性对照。基础往返可用，但自由探索范围仍有限。固定接受陌生信号更新没有可靠危险适应收益；扩大幅度明显损害正常搬运，直接方向修正也未解决问题。接受时机尚未在此蚁群任务中训练，旧方向小任务不能替代完整目标。

已可见结果仅作开发背景，见[初始试跑](docs/engineering/novel-adaptation-results.md)、[扩大幅度](docs/engineering/wider-update-results.md)、[方向修正](docs/engineering/novel-direction-adaptation-results.md)。不倒填为预先登记的分析，不把同巢蚂蚁、时间步或继承参数的多代当独立重复。

## Active Work

先执行已冻结的[候选更新短期价值诊断](designs/candidate-update-value.md)，最多 128 点、每分支 64 步。该诊断使用开发危险机制，不是未见环境优势检验；数据也不能冒充未接触危险的门控训练。训练期允许用分支后果构造监督，但部署只能看当前和过去。确认候选价值后再单独训练接受决策；全部跳过仅说明保护能力。随后补齐 MLP、纯规则及消融，达到同能力条件后冻结未见环境对照。总体[设计草案](designs/ant-self-training-adaptation.md)仍不可执行确认实验。

## Open Threads

反馈归因、接受决策训练目标、MLP 同能力资格、陌生环境范围和确认实验精度尚需落实。减速、周期及移动危险来源的完整对照尚未实现。个体终生内学习与死亡后参数继承分别报告，不把代际筛选替代自训练。

旧连续预测分支暂停，原设计未冻结、未执行，不判为证伪。旧状态原样保存于提交 `ceea87c`；数据库迁移另存于 `a382b08`。

## Key Decisions

每只模型的参数、记忆、随机流和优化状态独立；同组共享信息素，组间环境隔离且外部干预配对同步。规则组完全不用神经网络。基础能力、参数量、观测与计算预算同时披露。

顺序为推理、实际后果、更新提案、接受或跳过、下一次推理。基础动作冻结，保存固定间隔参数、残差、提案和行为记录。保留负差值，不预设自训练获胜。

停止当前扩大步长和局部结构变体。下一开发试验有明确预算与淘汰标准，正式留出不得用于调参。原 V2V 只读，不改变基础设施配置。

## Navigation

- [项目入口](README.md)
- [蚁群适应性对照设计](designs/ant-self-training-adaptation.md)
- [研究结构](research-tree/README.md)
- [基础往返结果](docs/engineering/colony-reward-results.md)
- [预算与行为审计](docs/engineering/budget-input-audit-results.md)
- [旧预测设计](designs/neural-readout-selective-rollback.md)

## References

结构化状态：`.research/research.sqlite`。适配共同基础：`logs/colony-reward/20260926T070617-2/episode-0008-ant-00.npz` 至 `episode-0008-ant-07.npz`。动作底座：`logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz`。开发运行目录、提交和局限由结果文档记录；这些模型未加入 Git，不以代码提交代替模型来源证明。
