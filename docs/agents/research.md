# 科研工作入口

## 当前边界

本项目的科研状态由 `RESEARCH.md` 与独立的 `.research/research.sqlite` 共同维护。当前主线为蚁群自训练的可行性和未见环境适应性，具体状态以科研首页与数据库为准。旧预测草案保留为暂停分支；已有开发试跑不等于完成确认性对照。

资料搬迁、环境检查与软件测试不等于科学实验。V2V 的原项目保持独立，本项目不得因阅读其材料而自动继承其研究结论或冻结记录。

## 开始工作

先加载项目级 `akira-research`，阅读 [科研首页](../../RESEARCH.md)，再执行：

```bash
bash scripts/research-db.sh status
bash scripts/research-db.sh research-tree
```

科研首页定位当前问题和边界；数据库保存对象及来源追踪。脚本只转交正式 Akira Research 命令行，不实现第二套数据库或自行修改 schema。

## 对象与目录

具体工作由 `akira-research` 路由到文献、假设、设计、数据、分析、解释或传播技能。只有真实科研对象形成后才建立相应目录和数据库对象，不用空目录或虚构的已完成记录充数。

`research-tree/README.md` 是数据库的派生视图。真实研究节点或关系改变后，通过以下命令更新，而不是手工改图制造关系：

```bash
bash scripts/research-db.sh render-research-tree-view
```

## 完成检查

每个独立科研状态变更先按 Guard 规则提交相关文件，再执行：

```bash
bash scripts/research-db.sh validate --completion
```

该检查证明当前已登记结构、来源和文件的一致性，不证明研究问题已解决或模型有效。发现历史数据库版本不兼容时先报告，不在只读任务中自动迁移。

## 数据与外部资料

赛事原始附件当前只是本机资料，见 [赛事资料](../competition/README.md)。真正用于研究前，由 `data` 明确来源、独立单位、范围和质量控制。教学记录与受限附件不得未经授权公开或传给第三方服务。

外部论文、旧项目结果、助手建议、用户决定与本项目新观察分别保留来源，不以聊天总结替代原文或真实实验记录。
