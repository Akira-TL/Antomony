# Study: 方向归一化方式与可塑连接学习配对实施

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/plastic-projection.md)

## Study Identity

实施`plastic-projection`已完成，批次为`20260927T000547-3`。真实开始为UTC 2026-09-27T00:05:47Z，依据脚本运行目录时间及执行会话67265；开始状态已在运行期间提交`604c9d2`。主线于UTC 2026-09-27T00:09:34Z取回退出码0，完成字段采用这个已观测返回时刻，不冒充精确进程退出时间。

## Source and Experimental Units

两个初始化1711/1712分别使用既有个体00/01方向底座及同一动作模型；冻结前已核对三份源参数散列。两模式的完成记录均声明两个初始化完整执行，样本按初始化登记并保存对应两模式目录。两个架构、训练前后及回合均在初始化内配对，不能当作独立模型。

## Actual Groups / Exposure / Intervention

按旧单位化模式`unit`再候选模式`bounded`真实执行；两者完成记录均声明固定终点训练与全部配对评价已执行。两模式按协议含训练前0轮、训练后150轮及学得接受、全部跳过、全部接受、等次数随机接受四组。当前已核对完成、代码与协议身份，尚未读取评价值或完成逐帧内容审计，不把完成声明当作效果通过。

## Sample Collection and Processing

两模式各两个初始化的原始记录已停止写入。主线独立固定完整目录清单，当前共460项且逐项散列核验通过；数据身份另行登记，不补采、不依据效果删回合。未重跑或挑选保存点。

## Assays and Measurements

按冻结协议产生96帧方向课程及结构保存点。完整测量配对和逐帧内容仍待正式分析审计；当前未读取正式评价结果。本课程没有物理搬运或死亡测量。

## Protocol / Materials / Instruments

执行协议为`.research/protocols/plastic-projection-unit.json`及`.research/protocols/plastic-projection-bounded.json`，两个实际协议副本已分别逐字节核对一致。两模式`commits.txt`均记录冻结提交`4de4323541da4e2dbff2170f6d7388a584ae6fb6`及代码身份`df7afbfd20bba451071e9cad429064c65e710cbb`。本任务只登记实施，不更改模型、协议、原V2V或8775。

## Batch / Run / Time

真实根目录为`logs/plastic-projection/20260927T000547-3`。主线实际命令为`env UV_CACHE_DIR=/tmp/mathhackson-uv-cache timeout --signal=KILL 900s bash scripts/training/plastic-projection.sh 4de4323541da4e2dbff2170f6d7388a584ae6fb6`。整批900秒硬限，单模式外部440秒强限、协议内部420秒软限。

旧模式内部计时101.25829220899323秒，候选模式99.55808637899463秒，分别来自各自`completion.json`。两文件修改时间分别为UTC 2026-09-27T00:07:35.857082162Z与2026-09-27T00:09:22.811356216Z；它们不是精确退出时点。主线取回整批退出结果的UTC 00:09:34Z与内部计时分别保留，不从目录时间或文件时间相减伪造精确整批耗时。

## Failures / Missing Events

主线实际退出码0，两模式完成字段均为`true`、完成初始化数均为2；未报告超时或补跑。完整配对、有限值和内容契约仍待核验，不能仅据完成文件排除所有数据缺失或行为失败。

## Deviations

当前已核验的协议、代码和完成身份未发现实施偏离；不将此表述扩大为效果有效或所有内容均已审计。后续若发现偏离，独立追加，不回写冻结设计。

## Outputs

完整原始目录由`data/plastic-projection/inputs.sha256`固定，共460项，清单自身SHA-256为`a83587452c069198b7d02ee3a57f9cc56844bdfbc62961759543a31068cac742`。清单由主线在退出且停止写入后独立生成，本任务复核散列；尚未进行效果汇总。根目录内两模式控制台日志随完整目录清单保留，数据身份随后独立登记。

## Record Boundary / Corrections

开始记录依据主线实际工具会话、脚本目录时间及本任务对目录存在性的核对，已在进程活跃时登记；完成依据真实退出结果及两模式完成文件，不回填精确退出时刻。身份与清单核验不读取评价值，实施完成不等于达到科研门槛。
