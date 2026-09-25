# Issue tracker: Local Markdown

本仓库没有远端，工程任务使用 `.scratch/` 内的本地 Markdown。该选择是新仓库初始化的本地默认，不代表已创建 GitHub Issues。

## Conventions

每个工作项使用 `.scratch/<feature-slug>/`；需要规格时为 `spec.md`。实现任务一项一文件，放在 `issues/01-<slug>.md`，从 01 编号，禁止汇总成一个无法独立领取的任务文件。

文件顶部使用机器可读 `Status:`，待分流请求同时使用 `Category:`。状态和类别通过 [角色映射](triage-labels.md) 解析。评论追加到 `## Comments`，依赖记为 `Blocked by: NN, NN`。

## Operations

“发布到 tracker”表示在对应目录建立文件，不创建外部资源。读取任务时使用完整相对路径；多个目录编号冲突时不猜测。先筛选开放、无阻塞且未领取的任务，再按当前流程次序执行。

普通任务领取以 `Status: claimed` 记录；只有全部验收条件均有证据时才改为 `Status: resolved`。多 Agent 并发领取必须交由明确的协调协议提供互斥，仅改文本不构成锁。

## Wayfinding

仅在显式进入该流程时使用 `.scratch/<effort>/map.md` 和各子任务。`Type: research|prototype|grilling|task`、`Status: claimed|resolved` 保留本地标准协议值，不生成 Wayfinder 远端标签。解决后在子任务写 `## Answer` 并把结论指针接回 map。
