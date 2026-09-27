# Study: 同一情境独立试行回报基线的配对实施

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/plastic-paired-baseline.md)

## Study Identity

实施`plastic-paired-baseline`正在进行，批次为`20260927T004157-3`。主线报告真实开始时间为UTC 2026-09-27T00:41:57Z，执行会话为60548。本记录只登记启动，不登记完成或效果判断。

## Source and Experimental Units

执行入口按冻结协议加载初始化1721/1722及各自的既有方向底座、同一动作模型。本任务已核对协议身份，未读取训练进度或效果文件，不预先登记任何初始化已经完成。实验单位、配对层级和来源材料以[冻结设计](../../designs/plastic-paired-baseline.md)为准。

## Actual Groups / Exposure / Intervention

主线已启动同时包含历史回报基线与同情境另一试行回报基线的执行脚本。当前只确认整批开始，不将两组均完成作为事实；实际组别完成情况须在取得退出记录后另行登记。科学规则与对照定义仅引用冻结设计，不在实施记录中重写。

## Sample Collection and Processing

原始记录正在运行目录中产生，尚未固定完整清单或进行数据整理。本任务不补采、不重启、不更改任何输入或原始记录。

## Assays and Measurements

测量内容与保存时点由冻结协议执行。当前未检查逐帧记录、完整性或奖励值，尚未登记已完成测量；本记录不提供效果证据。

## Protocol / Materials / Instruments

实际启动采用`.research/protocols/plastic-baseline-history.json`与`.research/protocols/plastic-baseline-paired.json`。主线报告代码提交为`be1386ce26fbb12801e20618b6878893f2b195d7`，冻结提交为`581a7b6378f9cb100bf876f4d0c554b130fd317d`；本任务已核对两个Git对象存在，数据库设计24所登记冻结点一致。原V2V和8775不在本次实施修改范围内。

## Batch / Run / Time

原始根目录为`logs/plastic-baseline/20260927T004157-3`，本任务已核对目录存在。主线报告实际命令为`env UV_CACHE_DIR=/tmp/mathhackson-uv-cache timeout --signal=KILL 780s bash scripts/training/plastic-baseline.sh 581a7b6378f9cb100bf876f4d0c554b130fd317d`。启动时间以主线真实执行记录为来源，不从目录修改时间推算；当前结束时间、退出码与最终耗时均待记录。

## Failures / Missing Events

待确认：本次登记时主线报告会话仍活跃，尚无退出结果。未读取效果或完整性记录，不能据此声称没有缺失、运行失败或行为失败。

## Deviations

当前启动事实未报告相对冻结设计的偏离。后续若发现偏离，应保留冻结设计并另行登记，不以修改原设计消除差异。

## Outputs

原始输出位置为上述运行根目录。由于仍可能写入，本次不固定数据散列、不登记完整数据集，也不启动结果分析；结束后由数据流程核对完整清单并建立数据身份。

## Record Boundary / Corrections

开始时间、命令、会话与活动状态来自主线的同时期执行报告；目录存在性、Git对象及数据库冻结身份由本任务机械核对。未读取正式效果，未操作或停止训练进程。以后补记结束信息时应明确来源，不把本次进行中记录改述为已经完成。
