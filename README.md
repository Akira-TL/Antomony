# MathHackson

Math Hackathon 2026 的独立工程与研究工作区。项目名称和路径按用户指定为 `/home/Akira/Projects/MathHackson`。

当前重点是蚁群模型能否学会控制在线参数更新，并在基础能力相当时适应未见环境。已有纯方向动作底座、局部信息素、多蚁独立模型及受限更新，尚未证明自训练优势。见[当前科研状态](RESEARCH.md)和[对照设计草案](designs/ant-self-training-adaptation.md)。旧预测草案保留为历史分支，不再作为当前规格。

## 阅读入口

| 入口 | 用途 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 唯一项目 Agent 约定与读写边界 |
| [CONTEXT.md](CONTEXT.md) | 已明确的项目术语与赛道对应关系 |
| [RESEARCH.md](RESEARCH.md) | 当前科研目标、问题与停止边界 |
| [研究结构](research-tree/README.md) | 由数据库生成的研究关系视图 |
| [单一机制提取](docs/competition/v2v-extraction-scope.md) | 提取内容、原实验边界、后续最小验证 |
| [蚁群适应性研究](designs/ant-self-training-adaptation.md) | 同能力对照、更新时机与陌生环境；尚未冻结 |
| [旧演示与开发](docs/competition/track2-demo-development.md) | 早期预测方案，仅作历史记录 |
| [旧预测设计草案](designs/neural-readout-selective-rollback.md) | 历史分支，未冻结未执行 |
| [赛事资料](docs/competition/README.md) | 赛题、原始附件位置、历史说明与数据约束 |
| [技能说明](docs/agents/skills.md) | Matt / Research 项目引用、来源版本与恢复 |
| [工程任务约定](docs/agents/issue-tracker.md) | 本地任务、依赖与领取规则 |
| [科研操作说明](docs/agents/research.md) | 正式科研数据库命令和完成检查 |

## 开发与研究命令

在本项目根目录执行：

```bash
cd /home/Akira/Projects/MathHackson

# 创建或同步 Python 3.12 环境；依赖与版本范围以 pyproject.toml / uv.lock 为准。
uv sync --locked

# 恢复项目级技能引用。要求本机已经安装清单中对应版本的受管技能。
bash scripts/setup-skills.sh

# 基础测试及项目技能来源检查；日志在 logs/check.log。
bash scripts/check.sh

# 查看科研数据库与当前研究结构。
bash scripts/research-db.sh status
bash scripts/research-db.sh research-tree

# 在提交科研状态后执行完整性检查。
bash scripts/research-db.sh validate --completion

# 查看 CodeGraph；代码变化后同步索引，再查询符号。
codegraph status --json
codegraph sync .
codegraph explore 'load_skills'
```

当前包位于 `src/mathhackson/`，依赖以 `pyproject.toml` 和 `uv.lock` 为准；训练使用 PyTorch，测试使用 pytest，不依赖原 V2V 运行。`FastResidualParameter` 与 `RollbackRequest` 是参数操作接口，不等于学习策略有效。运行入口见[蚁群运行说明](docs/engineering/ant-colony-run.md)。

## CodeGraph

本项目已用本机 CodeGraph 执行 `codegraph init .` 并验证符号查询。`.codegraph/` 是可重建索引，不进入 Git；它只扫描本项目受支持的代码，未把原始教学材料或外部技能引用作为项目源码。

在新 checkout 首次使用时运行 `codegraph init .`。此处不自动安装或更新全局 CodeGraph，也不修改 Codex 等执行器的全局配置。项目代码尚少时，索引规模仅反映维护工具，不代表已经存在研究模型。

## 目录职责

```text
.agents/skill-view.tsv       项目技能来源清单，进入 Git
.agents/skills/              指向机器级技能的本机引用，不进入 Git
.research/research.sqlite    独立科研数据库，进入 Git
research-tree/              数据库生成的人类研究关系视图
materials/competition/      原始附件、解压资料与历史说明
materials/web-snapshots/    已迁移的网页快照，仅本机保存
docs/agents/                工程与科研入口约定
docs/competition/           赛事资料索引与迁移清单
scripts/                    维护及后续运行入口
tests/                      工程测试
src/mathhackson/            独立提取的可逆参数更新实现
.scratch/                   正式本地工程任务
logs/                       运行日志，不进入 Git
```

研究目录按真实对象逐步建立，不预生成空的 literature、hypotheses、designs、analysis 或 communication 目录。

## 资料与独立项目边界

[迁移清单](docs/competition/materials-manifest.csv) 记录了从 Downloads 移入的全部 29 个资料文件及 SHA-256。原始附件、解压数据和网页快照保留在本机并排除出 Git；历史说明原样保留，其中旧路径和早期推荐的适用边界见赛事资料入口。

Windows 下载目录中的原始下载副本保持原样，不在本次迁移范围。本仓库不自动公开或提交教学数据。

V2V 仍位于 `/home/Akira/Projects/v2v`，保持只读。本仓库仅从明确提交提取一个参数模块及其测试，来源见 `docs/competition/v2v-source-manifest.csv`。提取后的边界修复只发生在本仓库；没有复制完整模型、研究数据库、检查点或实验结果，也不改变原项目的科研状态。

当前 Git 分支为 `main`，未配置远端。正式提交使用 Akira Guard；远程仓库创建、推送与发布须另有用户授权。
