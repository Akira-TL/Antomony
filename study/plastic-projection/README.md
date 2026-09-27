# Study: 方向归一化方式与可塑连接学习配对实施

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/plastic-projection.md)

## Study Identity

实施`plastic-projection`真实启动，状态`in_progress`；批次为`20260927T000547-3`。主线报告UTC 2026-09-27T00:05:47Z启动，依据脚本生成的运行目录时间及仍活跃的执行会话67265；它不是预填的计划时间。当前仅登记已发生的开始，不填写完成时刻或退出码。

## Source and Experimental Units

计划两个初始化1711/1712，分别使用既有个体00/01方向底座及同一动作模型；冻结前已核对三份源参数散列。两个架构、训练前后及回合均在初始化内配对，不能当作独立模型。当前不把尚在产生的样本标成已完成。

## Actual Groups / Exposure / Intervention

已核验实际根目录含`unit/`及正在写入的`unit.log`，当前没有据此声称候选模式已启动。设计规定按旧单位化模式再候选模式执行，两者均含训练前0轮和训练后150轮，以及学得接受、全部跳过、全部接受和等次数随机接受；这些仍是当前后续计划，不冒充已完成组别。

## Sample Collection and Processing

真实进程正在产生记录。当前没有根据部分输出登记完整样本、数据对象或清单；完成后按实际模式和初始化记录完整、失败或缺失，不补采、不依据效果删回合。

## Assays and Measurements

按冻结协议记录96帧方向课程及结构保存点。尚未核验完整测量覆盖，未读取正式评价结果；本课程没有物理搬运或死亡测量。

## Protocol / Materials / Instruments

执行协议为`.research/protocols/plastic-projection-unit.json`及`.research/protocols/plastic-projection-bounded.json`，冻结提交`4de4323541da4e2dbff2170f6d7388a584ae6fb6`；启动代码身份为`df7afbfd20bba451071e9cad429064c65e710cbb`。本任务只登记实施，不更改模型、协议、原V2V或8775。

## Batch / Run / Time

真实根目录为`logs/plastic-projection/20260927T000547-3`。主线实际命令为`env UV_CACHE_DIR=/tmp/mathhackson-uv-cache timeout --signal=KILL 900s bash scripts/training/plastic-projection.sh 4de4323541da4e2dbff2170f6d7388a584ae6fb6`。整批900秒硬限，单模式外部440秒强限、协议内部420秒软限；当前执行会话仍活跃，不从计划上限推算结束时刻。

## Failures / Missing Events

尚未获得退出记录，不能声明完整完成或没有失败。若发生超时、退出异常或缺失，保留原件并记录，不补点。

## Deviations

目前只核验真实启动，没有确认的实施偏离；这不代表后续完整运行已通过核验。任何实际偏离另记，不回写冻结设计。

## Outputs

当前原始输出仍在根目录产生，不能固定完成清单或开展效果分析。全部文件停止写入后另行核验并交数据登记；目录外控制台记录与数据目录范围分别说明。

## Record Boundary / Corrections

开始记录依据主线实际工具会话、脚本目录时间及本任务对目录存在性的核对；当前没有声称精确纳秒启动、退出或运行耗时。此次在进程活跃时登记开始，而不是等结束后倒填。完成与分析结果须独立后续登记。
