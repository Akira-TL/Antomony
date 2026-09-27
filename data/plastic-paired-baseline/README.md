# Dataset: 同情境独立试行回报基线的原始记录

## Navigation

- [研究首页](../../RESEARCH.md)
- [实际实施](../../study/plastic-paired-baseline/README.md)
- [冻结设计](../../designs/plastic-paired-baseline.md)

## Dataset Identity

数据身份为`plastic-paired-baseline`，来源批次为`20260927T004157-3`。两组退出后固定508项原件，UTC 2026-09-27T00:52:38Z完成接收和全部散列复核；未读取评价值。

## Research Purpose

用于冻结设计24规定的两种回报基线、训练前后及四种更新方式配对比较。科学规则只引用[冻结设计](../../designs/plastic-paired-baseline.md)，本数据记录不修改估计目标、窗口或门槛。

## Source and Version

来源为实施22，原始目录`logs/plastic-baseline/20260927T004157-3`。历史组记录代码`be1386ce26fbb12801e20618b6878893f2b195d7`，候选组记录代码`27f120521ebe2c2d49fc6f5ea582a108e96efba2`，共同冻结点为`581a7b6378f9cb100bf876f4d0c554b130fd317d`。两个提交之间`src/`和`scripts/`无差异；两组实际协议副本均与对应冻结协议逐字节一致。主线于UTC00:48:21取回整批退出码0，两组均声明两个初始化完整执行。

## Population and Sample Mapping

独立外层初始化为1721和1722，分别对应每组的`seed-1721`与`seed-1722`目录。未经训练的参照位于各组`untrained/`下同名目录。两基线、训练前后、同情境两次试行、条件、评价回合和更新方式均在初始化内配对或嵌套，不能作为独立模型计数。

## Data Layers and Artifacts

原始层包含实际协议、代码身份、完成记录、逐轮日志、结构保存点与训练和评价轨迹，保留在本地忽略目录，不复制或覆盖原件。完整清单为`inputs.sha256`，共508项；派生结果由后续分析生成，不混入原始输入。

## Metadata / Missingness / Exclusions

组别、初始化、结构版本、条件、回合、四帧判断以及原始参数身份均须由后续审计核对。不按奖励大小剔除文件或回合；目前未完成完整性检查，不能声称没有缺失。

## QC and Anomalies

代码身份、实际协议及508项原件散列已核对，尚未读取评价值或完成逐帧审计。保存点、有限值、配对、实际奖励和在线写入约束须在正式分析中核对；实施退出及散列通过不等于内容质量或效果通过。

## Processing and Reproduction

生成入口为`scripts/training/plastic-baseline.sh`。主线在停止写入后用项目维护的清单入口固定完整输入，本任务以`sha256sum --check --quiet data/plastic-paired-baseline/inputs.sha256`复核通过；未修改原始值、补采、重跑或挑保存点。

## Freeze / Access / Ethics

完整清单SHA-256为`9d584a78d0c468fc2917a45f3aaea903ad8bdf573d16d7d699d01ced4f6297b6`，原件在首次汇总前固定。分析计划成功登记后才允许计算结果；这不将探索性比较变成确认性研究。数据仅来自本地合成课程，无真实生物或个人资料，不公开大体积原始目录，不修改原V2V及8775。
