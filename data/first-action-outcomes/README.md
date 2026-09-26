# Dataset: 首步方向分支数据

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/first-action-outcomes/README.md)

## Dataset Identity

`first-action-outcomes`，待真实生成后登记，当前不是已接收数据。

## Research Purpose

仅回答首次受伤前单次方向选择的32步后果，不训练。

## Source and Version

来源 `window-cadence` 的 `20260926T165559-3/four`，冻结配置保存文件清单与协议散列；源模型由原执行元数据核对。

## Population and Sample Mapping

种子19701、19702；点身份为种子、干预前时点、焦点编号；每点方向0至15。

## Data Layers and Artifacts

计划原始目录 `logs/first-action-outcomes/A001`；生成后以 `manifest.sha256` 固定，不覆盖原记录。

## Metadata / Missingness / Exclusions

最多8点，不足不补；超时和重建失败保留部分文件，不作完整结果。

## QC and Anomalies

原输入身份、父轨迹和快照须核验；分支逐帧累计值须与点记录一致。

## Processing and Reproduction

原始世界完整状态经重建复制隔离；参数与场快照不被宣称为可独立恢复世界。专用入口记录来源提交与运行时间。

## Freeze / Access / Ethics

本地模拟，无受试者。原始大文件忽略保存；未生成文件不提前登记为存在。
