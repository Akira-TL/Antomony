# v4制作记录与PR交付说明

## 本版内容

大纲在新动画实现之前提交，见`../outline.md`。英文画面和旁白，外置英中字幕，一条字幕只有两行：英文在上、中文在下。主片不烧录或嵌入任何字幕轨。

主线增加了最新比赛切片中的资格迹、带符号调制、每四个仿真行动一次的判断机会，以及外层更新规则训练与运行中局部适应的区别。旧的撤回介绍不占据主线。原v2/v3源码和主片逐字节保护。

所有数学和运动示意重新渲染；蚂蚁复用已认可的六腿矢量资产、真实转向和路径驱动步态。两只蚂蚁的漂移路径沿用明确标注的教学模拟，不拿它冒充新可塑方向模块在物理世界里的成绩。新资格迹、调制矩阵画成低维局部示意，没有流程框或全套V2V架构。

## 研究与语义

- 主仓库的读取提交、文件路径和内容散列见`../provenance.json`。
- 局部资格迹使用外积及衰减，输出侧采用已选方向与先验方向均值的差。调制信号有正、零、负，随后进行单步和累计权重限幅。
- 一次候选可以跳过；跳过保持F不变，但不冻结资格迹或反馈历史。判断周期的单位是四次仿真行动，不是四个视频帧。
- φ在跨情境课程中训练，外层训练可以使用反向传播；运行期局部改变的是F。不能把本片说成所有训练都没有梯度。
- 02:26后的记录结果来自已完成continuous-adaptation批次，而非新plasticity基础课程：四世界种子、三复杂条件等权，一组八个模型初始化。学得接受相对跳过交付+5.25，相对固定接受−1.00；固定接受相对跳过+6.25为相同均值差推得。不是显著性结论。危险条件学得接受比固定接受多1只死亡，不能宣称全面更好。
- 最新paired-baseline课程完成训练，但效果分析被有效性审计阻断，尚无效应估计。片中没有借此制造新胜负。
- 实时镜头是动作读取已提交参数、更新服从预算的设计目标，不声称已有确定毫秒数或硬实时保证。

## 重建命令

```bash
bash scripts/video/progress/run.sh audio
bash scripts/video/progress/run.sh still all draft
bash scripts/video/progress/run.sh render all hd
bash scripts/video/progress/run.sh assemble
bash scripts/video/progress/run.sh check
bash scripts/video/progress/run.sh review
```

可选画面检查：

```bash
uv run --no-project --python 3.12 --with numpy --with pillow -m communication.video.progress.production.qa --frames --final
```

交互式局部重播：

```bash
grep -n 'checkpoint:' communication/video/progress/scenes.py
bash scripts/video/progress/run.sh interactive C03Eligibility draft LINE_NUMBER
```

在ManimGL的交互终端中，对选中复制的同一checkpoint调用`checkpoint_paste()`即可复播。正式导出使用同一场景源代码。

## 输出

全部新媒体只写入`communication/video/rendered/v4/`；日志在`logs/video-v4/`。

- `Learning_While_Acting_v4_1080p60.mp4`：英文音轨的干净主片。
- `Visuals_v4_silent.mp4`：同一画面，无任何音轨和字幕轨。
- `Narration.en.wav`：48kHz单声道完整英文旁白，180秒。
- `Learning_While_Acting_v4.en-zh.srt`：UTF-8带BOM，每条两行、英文在上。
- `.en.srt`及`.zh.srt`：与双语稿相同时码的独立单语字幕。
- `index.html`：章节审片页，字幕预览默认关闭；开启时只是网页覆盖层，不修改视频字节。
- `ManimGL_sources_v4.zip`：代码、大纲、旁白和来源散列；不打包字体、第三方视频或模型文件。

源代码包依赖原教学模拟的`toy-gate.npy`。现有工作区已有该文件；全新环境应先用原`prepare-toy.sh`重建教学资产，不能用另一研究模型悄悄替代。TTS使用原来已采用的Microsoft en-US-AndrewNeural，不克隆3b1b声纹。

## 已执行与待核验

大纲和配音先行；49个双语句对逐一对齐到TTS句界，短句经有限速度调整与留白排进180秒，未截断台词。十幕末状态均在ManimGL中执行并查看合成联系表；发现的坐标类型和场景发现问题已修复。正式1080p60渲染另行执行。

最终是否通过帧数、时长、完整解码、字幕两行及原版保护，以生成的`render-manifest.json`、`qa/structural-checks.json`和`delivery-receipt.json`为准；不以本文替代运行结果。实际画面审查是关键时点抽样，不冒充逐帧人工检查或独立观众评测。
