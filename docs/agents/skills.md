# 项目级技能

## 范围

本项目显式引用 44 个已安装技能：Matt 稳定 engineering/productivity 及两个 Parallel 扩展共 28 项；Akira Research series 共 13 项；通用 `akira`、`akira-guard`、`browser-access` 共 3 项。

科研评议 Review series 与 Research series 平级，不在此次项目级引用范围；实际需要评议时再通过 `akira` 加载或补齐。不因安装 Parallel 扩展而自动启动其他 Agent。

## 安装与版本

本机已有上述技能，且 Akira 安装器的 `doctor` 检查通过。按 Akira 安装契约，真实内容继续来自机器级共享 Git checkout，项目不复制技能，也不修改全局注册表：

```text
项目 .agents/skills/<name>
    -> ~/.agents/skills/<name>
    -> ~/.agents/sources/<owner>/<repo>/<source_path>
```

本次实际引用的仓库为 `Akira-TL/matt-skills`、`Akira-TL/akira-research-skills` 与 `Akira-TL/skills`。精确来源、ref、commit 和源内路径以 `.agents/skill-view.tsv` 为准。

项目级引用按用户明确要求建立，不是安装器自动投影。`.agents/skills/` 是本机可重建视图，排除出 Git；来源清单和建立脚本进入 Git。

## 2026-09-27版本核对

研究技能的13条来源由 `6b11f8014b0856ede77719716504b087bd53a828` 显式更新为 `b1491649650e7bda9c57d7c3d6411570c118771e`。另外31条来源、项目链接与机器级注册保持不变，未复制或重新安装技能。

修复前，共享研究仓库已经处于 `1332d61`，比旧清单多出8个本地提交：`ef660f7`、`7305127`、`90bbb64`、`abbae43`、`8e0780a`、`53a1d24`、`5f8e054`、`1332d61`。这些是先前已有状态，不是本轮新增；差异涉及按分析模式区分估计结果门禁、历史数据与分析文件的兼容范围、运行时软链接、文献候选阻塞与延期、数据集与论文关联，以及人类可读资料路径的显式迁移。

本轮新增的 `b149164` 仅修正纯机器散列日志被当作科研正文的问题：全文只有逐行 `字段=40位或64位十六进制散列` 的 `.txt` 记录保留技术字段；混入正文、注释、任意文本或不完整散列时继续检查，Markdown 不适用这项识别。该修复不改写既有证据、研究状态或结果，也不按附件角色整体豁免材料。

版本核对时，本项目数据库的 `PRAGMA user_version` 与 `meta.schema_version` 均为29，与共享技能当前结构版本一致，必需表完整，包括 `dataset_papers`。本次只读核对结构，不执行数据库迁移或人类资料路径迁移，不据此宣称科研完成门禁或模型效果通过。项目命令仍通过既有机器级链接加载共享技能；固定版本清单更新只是如实记录已核对的来源。

## 恢复和校验

在已有相同机器级技能的电脑上：

```bash
bash scripts/setup-skills.sh
bash scripts/setup-skills.sh --check
```

脚本逐项验证来源仓库、源内路径和 Git 版本，先全部预检再建立软链接；已有不同内容会报错，不覆盖。机器级技能升级后，清单不静默跟随，必须复核差异并显式提交新清单。

缺少机器级技能时，先阅读已安装 `akira` 的 `references/CATALOG.md` 与 `references/INSTALLATION.md`，按正式 `skills.py` 安装入口补齐，再建立项目引用。本项目脚本不自动联网安装或更新机器级能力。

## 入口

工程工作进入 `ask-akira`；普通模式由它路由到 `ask-matt`，比赛任务按其比赛模式选择当前切片。研究工作进入 `akira-research`，先读 `RESEARCH.md` 和数据库状态。技能缺失或版本不一致时先解决能力边界，不从旧会话重造技能内容。
