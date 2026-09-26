# Dataset: 四步与十六步窗口轨迹与参数

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/window-cadence/README.md)

## Dataset Identity

`window-cadence`，版本 `20260926T165559-3`，全部16世界。

## Research Purpose

验证缩短固定窗口是否减少实际失败，不以更多判断或参数写入替代学习效果。

## Source and Version

本机代码 `4410980` 生成，接收时间2026-09-26T16:57:05.295774+00:00，模型来源由执行记录散列固定。

## Population and Sample Mapping

19701、19702各两窗口四组，每组8只2048步。单位为世界种子，个体、生命、时间及同种子配置相关。

## Data Layers and Artifacts

原始数据 `logs/window-cadence/20260926T165559-3/`，完整[文件清单](manifest.sha256)冻结身份；分析派生结果另存，不覆盖原始记录。

## Metadata / Missingness / Exclusions

无剔除。保存原始奖励及实际生存反馈、伤害、终止与复活、每次判断和周期参数快照；不补造未保存的全场信息素。

## QC and Anomalies

完整核验通过：4470文件、32768帧、20652候选时序及1632记忆快照，三个无更新对照跨配置行为一致。无剔除或原始数据修改；原始计数在正式分析登记前可见，不宣称结果盲化或外部预注册。

## Processing and Reproduction

`scripts/training/window-cadence.sh` 生成，原始文件不筛选；完整散列清单与来源先提交，分析另行登记执行。

## Freeze / Access / Ethics

本机模拟，无受试者，不外传。大型参数和轨迹不加入Git，配置、来源与完整清单进入Git。
