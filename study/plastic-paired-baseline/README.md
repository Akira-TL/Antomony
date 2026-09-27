# Study: 同一情境独立试行回报基线的配对实施

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/plastic-paired-baseline.md)

## Study Identity

实施`plastic-paired-baseline`已完成，批次为`20260927T004157-3`。真实开始为UTC 2026-09-27T00:41:57Z，执行会话60548；进行中记录已提交`27f1205`。主线于UTC 00:48:21取回退出码0，完成字段采用这个实际观测返回时刻，不冒充精确进程退出时间。

## Source and Experimental Units

两个初始化1721/1722按实际目录登记。两组完成记录均声明两个初始化完整执行；本任务核对完成与协议身份，未读取效果文件。实验单位、配对层级和来源材料以[冻结设计](../../designs/plastic-paired-baseline.md)为准，不能把组别或回合当作独立训练模型。

## Actual Groups / Exposure / Intervention

按历史回报基线组再同情境另一试行基线组执行，两组完成字段均为`true`。科学规则与对照定义仅引用冻结设计，不在实施记录中重写；完成声明不代表逐帧内容或效果已经通过审计。

## Sample Collection and Processing

主线在退出、停止写入后独立固定完整清单，共508项，本任务逐项散列复核通过。未补采、重启或更改任何输入及原件，数据身份随后独立登记。

## Assays and Measurements

完成记录声明固定终点训练与全部配对评价已执行。测量内容与保存时点由冻结协议执行；本任务尚未读取奖励值或检查逐帧内容，不把完成声明和散列通过解释为效果证据。

## Protocol / Materials / Instruments

实际启动采用`.research/protocols/plastic-baseline-history.json`与`.research/protocols/plastic-baseline-paired.json`，原始协议副本分别逐字节核对一致。历史组实际代码为`be1386ce26fbb12801e20618b6878893f2b195d7`，候选组为`27f120521ebe2c2d49fc6f5ea582a108e96efba2`；共同冻结提交为`581a7b6378f9cb100bf876f4d0c554b130fd317d`。两组之间发生启动记录提交，已核对`src/`及`scripts/`无差异，不把不同提交号误述为不同执行实现。原V2V和8775未修改。

## Batch / Run / Time

原始根目录为`logs/plastic-baseline/20260927T004157-3`。实际命令为`env UV_CACHE_DIR=/tmp/mathhackson-uv-cache timeout --signal=KILL 780s bash scripts/training/plastic-baseline.sh 581a7b6378f9cb100bf876f4d0c554b130fd317d`。历史组内部169.6911693459988秒，候选组170.05527329799952秒，分别来自完成记录。

主线报告两完成文件修改时间分别为UTC 00:44:54.849371396与00:47:54.467390241；它们不是精确退出时刻。整批退出码0于00:48:21取回，不从目录或文件时间推算精确进程耗时。

## Failures / Missing Events

整批退出码0，两组均声明完整执行两个初始化；未报告超时或补跑。尚未读取正式效果和完成逐帧审计，不能仅据退出码排除内容缺失或行为失败。

## Deviations

已核对的协议、执行实现与完成身份未发现相对冻结设计的偏离；两组代码提交号不同仅涉及中间登记实施事实。后续若发现其他偏离，另行登记，不修改冻结设计。

## Outputs

完整原始清单为`data/plastic-paired-baseline/inputs.sha256`，共508项，清单SHA-256为`9d584a78d0c468fc2917a45f3aaea903ad8bdf573d16d7d699d01ced4f6297b6`。主线在退出后独立生成，本任务复核全部散列通过，尚未读取评价值或汇总效果。

## Record Boundary / Corrections

开始记录在进程活跃时独立提交，真实结束依据主线取回退出码及两组完成文件另行补记。目录、代码身份、实际协议和清单由本任务机械核对；未读取正式效果，未操作或停止训练，实施完成不等于达到科研判据。
