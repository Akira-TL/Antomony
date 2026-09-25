# MathHackson

Math Hackathon 2026 的独立工程与研究工作区。项目名称和路径按用户指定为 `/home/Akira/Projects/MathHackson`。

当前完成的是资料迁移、工程规范、项目级技能引用与科研基础；最终赛道和研究问题尚未确定，尚未开展模型开发、实验、训练或电磁仿真。

## 阅读入口

| 入口 | 用途 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 唯一项目 Agent 约定与读写边界 |
| [CONTEXT.md](CONTEXT.md) | 已明确的项目术语与赛道对应关系 |
| [RESEARCH.md](RESEARCH.md) | 当前科研目标、问题与停止边界 |
| [研究结构](research-tree/README.md) | 由数据库生成的研究关系视图 |
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

当前开发环境没有第三方 Python 模型或应用依赖，只有实际使用的维护脚本与对应测试。正式选择研究对象后再添加必要依赖；不预装 PyTorch、前端框架或电磁软件。

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
src/                        选题后按需要建立真实模型或应用模块
.scratch/                   正式本地工程任务
logs/                       运行日志，不进入 Git
```

研究目录按真实对象逐步建立，不预生成空的 literature、hypotheses、designs、analysis 或 communication 目录。

## 资料与独立项目边界

[迁移清单](docs/competition/materials-manifest.csv) 记录了从 Downloads 移入的全部 29 个资料文件及 SHA-256。原始附件、解压数据和网页快照保留在本机并排除出 Git；历史说明原样保留，其中旧路径和早期推荐的适用边界见赛事资料入口。

Windows 下载目录中的原始下载副本保持原样，不在本次迁移范围。本仓库不自动公开或提交教学数据。

V2V 仍位于 `/home/Akira/Projects/v2v`，保持独立；本仓库没有复制其代码、模型、研究数据库或实验结果，也不修改它的科研状态。

当前 Git 分支为 `main`，未配置远端。正式提交使用 Akira Guard；远程仓库创建、推送与发布须另有用户授权。
