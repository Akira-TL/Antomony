# Domain Docs

## Before exploring, read these

本仓库采用 single-context，领域词汇以根目录 `CONTEXT.md` 为准。探索相关代码前先读其适用条目；架构决策存在时读取相应记录。

## ADR convention

初始化前没有 ADR。后续真正出现难以逆转、需要解释的架构权衡时，使用 `docs/adr/NNNN-slug.md`，按 `domain-modeling/ADR-FORMAT.md` 创建。当前没有需要记录的架构决定，不预造空 ADR 或多上下文结构。

## Vocabulary

项目术语使用 `CONTEXT.md` 中已明确的含义。未决定的研究方向不写成已确立架构；一般编程概念不扩展为项目专用术语。

## Conflicts

新方案与既有 ADR 冲突时明确指出，并由用户或相应流程决定是否重新讨论，不静默覆盖历史决定。
