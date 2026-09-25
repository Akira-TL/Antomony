# Workflow Role Mapping

本地 Markdown 的状态与类别映射如下；无需维护远程标签注册表。

| Workflow role | Value in our tracker | Used by | Meaning |
| --- | --- | --- | --- |
| `ready-for-agent` | `ready-for-agent` | `to-tickets`, `triage` | 条件明确，可由 Agent 执行 |
| `bug` | `bug` | `triage` | 缺陷请求 |
| `enhancement` | `enhancement` | `triage` | 新功能或改进 |
| `needs-triage` | `needs-triage` | `triage` | 待评估 |
| `needs-info` | `needs-info` | `triage` | 等待必要信息 |
| `ready-for-human` | `ready-for-human` | `triage` | 需要人的判断或操作 |
| `wontfix` | `wontfix` | `triage` | 不执行 |

普通实现任务的 claimed/resolved 与 Wayfinder 的 Type/生命周期字段按 owner 流程处理，不把本地协议字段伪装成远端标签。
