# Study: 近似规模模型训练及基础轨迹

## Navigation

- [研究首页](../../RESEARCH.md)
- [冻结设计](../../designs/matched-foundation.md)

## Study Identity

批次 `20260926T121232-2`，八个独立模型各2400轮，48个评价世界完成。
设计提交 `5e239f0`，成功登记为Design 11后提交 `e6e36ac`，正式运行晚于该提交。

## Source and Experimental Units

模型种子81至88，评价世界17101至17108。原14宽参数来源与动作来源共九个文件，身份及散列在 `sources.jsonl`。同巢个体和食物/空场两个任务相关，世界种子是独立推断单位。

## Actual Groups / Exposure / Intervention

新17宽模型从独立初始化训练，只有信号课程；原14宽模型及纯规则不训练。评价三组均运行食物和无食物任务，组间信息素、库存、个体隔离。使用有限巢穴源及局部平滑沉积，没有复杂来源。

## Sample Collection and Processing

按固定顺序运行48世界，未排除、补种子或挑选中途模型。每200轮保存，固定2400轮参数重新加载后参与评价；所有评价更新次数为零。源模型与最终参数的散列核对未变。逐世界输出已可见，后续分析不称盲法。

## Assays and Measurements

104份参数、96份优化器状态、96条训练诊断，以及48份压缩轨迹。世界记录含真实拾取、交付、预算返回、耗尽与最大半径。完整性和汇总值由后续分析核对；完成执行不等于达到资格。

## Protocol / Materials / Instruments

协议 `.research/protocols/matched-foundation.json`；入口 `scripts/training/matched-foundation.sh`；执行提交 `e6e36acf3131aa48d9c8f1e46eff4e885b01ee7f`。项目锁定环境，PyTorch数值线程数1；每只独立优化器及数据随机流，保留训练课程和超参数。

## Batch / Run / Time

原始目录 `logs/matched-foundation/20260926T121232-2/`。开始2026-09-26 12:12:32.738871 UTC，完成12:14:15.961149 UTC，约103.22秒，不含工程开发和分析耗时。

## Failures / Missing Events

进程退出0，48世界输出完整，未发生运行失败。表现不佳或耗尽不属于排除理由，仍完整保留。

## Deviations

不适用：未发现相对冻结设计的实施偏离。17999短流程独立保留，未混入正式批次。

## Outputs

253份原始文件：104参数、96优化器、48轨迹及配置、执行元数据、来源、训练诊断和世界记录五个文件。以完整散列清单交给数据登记，不改写原始产物。

## Record Boundary / Corrections

上述为执行日志与产物的直接记录，不构成能力等效、自训练可行性或复杂环境适应优势的证据。评价全程冻结，后续在线效果仍须完整连续对照。
