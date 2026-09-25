# MathHackson 项目约定

## 工作边界

本仓库是数学建模黑客松的独立工程与研究工作区。先读 `README.md` 与 `CONTEXT.md`；选择赛道和研究范围时读 `docs/competition/README.md`。本轮初始化不代表用户已经选定赛道。

`/home/Akira/Projects/v2v` 是独立参考项目，只读；未经用户新授权，不移动其代码、数据库或模型，不启动其训练，不修改已冻结的设计和判据。

## Agent skills

### Issue tracker

工程需求和任务使用本地 `.scratch/<feature>/`。创建、领取或完成任务之前读 `docs/agents/issue-tracker.md`。

### Workflow roles

任务状态与类别沿用标准角色值；分流和发任务之前读 `docs/agents/triage-labels.md`。

### Domain docs

本仓库采用 single-context。领域词汇或架构决策变化前读 `docs/agents/domain.md`。

### Engineering

软件工程入口为 `ask-akira`；模式由当前任务决定，普通任务进入标准 Matt 流程。项目级技能引用及恢复方式见 `docs/agents/skills.md`。

### Research

研究入口为 `akira-research`。先读取 `RESEARCH.md`，再使用 `bash scripts/research-db.sh status` 核对数据库；只读任务不得写入数据库。用户现已进入 V2V 单一机制提取阶段；当前范围见 `docs/competition/v2v-extraction-scope.md`。只在本仓库隔离和验证选定的小模块，原 V2V 仍只读；新的科研实验须先固定研究设计，不把工程测试等同于效果证据。

## 资料与网络

原始附件、解压教学数据和网页快照存于 `materials/` 的忽略目录。其完整性由 `docs/competition/materials-manifest.csv` 记录；原件不改写，不强制加入 Git，不未经授权外传。科研数据真正进入分析前，另按 `data` 的规范登记来源与范围。

本任务不修改网卡、路由、防火墙、代理、nginx 或 ForgeRelay 服务，保持远程连接不受项目操作影响。

## 开发执行

Python 通过 `uv` 与项目 `.venv` 运行；检查和后续运行入口放在 `scripts/*.sh`，日志写入 `logs/`。研究模型、前端和仿真依赖按正式选题再添加，不在基础工程中预设。

若 `.codegraph/` 已存在，定位或理解代码先运行 `codegraph explore`，再按需要读源码。只索引本项目代码，不跨目录扫描 V2V、技能源或受限数据。

每个独立修改目的按全局 Guard 规则分别提交；不创建远程仓库、不推送、不发布，除非用户明确授权。`AGENTS.md` 是唯一项目指令来源，不维护另一份执行器专用副本。
