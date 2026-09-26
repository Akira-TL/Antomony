# Learning While Acting：动态蚂蚁版

## 当前版本

本版以持续运动的关节蚂蚁、相同扰动下的轨迹对照、参数数轴、局部预测图和后果时间关系解释自学习。制作约定见[重制镜头说明](visual-redesign-v4.md)，配音见[英文旁白](narration-v4.en.txt)。上一版的静态流程框、章节开场和烧录字幕不再使用。

完整生成命令：

```bash
bash scripts/video/prepare-toy.sh
bash scripts/video/audio.sh
bash scripts/video/retime-audio.sh
bash scripts/video/render.sh render all hd
bash scripts/video/assemble.sh
bash scripts/video/open-review.sh
```

`hd`为1920×1080、60fps；`draft`为1280×720、30fps。总时长180秒。输出在`communication/video/rendered/v2/`，其中`Learning_While_Acting_v2_1080p60.mp4`仅有画面与旁白，`Learning_While_Acting.en.srt`为独立文件，不烧录也不嵌入视频。`index.html`提供九段跳转，字幕不会默认显示。

## 单幕调试与检查点

```bash
bash scripts/video/render.sh preview A04LearnFromTheGap draft
bash scripts/video/render.sh render A05NoiseOrChange hd
```

源场景在`scenes/ants_learning.py`。查询需要停下的检查点行号：

```bash
grep -n 'checkpoint:' communication/video/scenes/ants_learning.py
```

用实际行号进入ManimGL交互环境：

```bash
bash scripts/video/render.sh interactive A04LearnFromTheGap draft LINE_NUMBER
```

在编辑器复制一个以稳定注释开头的动画片段，然后在交互终端调用`checkpoint_paste()`。可反复回到相同检查点重放；`checkpoint_paste(skip=True)`看末状态，`checkpoint_paste(record=True)`记录小段。单Scene交互主要检查图形；完整配音由后续合成对齐。

`film_plan.py`保存九段时长和旁白句子出场时间。不要用整段画面速度调整来迁就配音，也不要重新加入字幕渲染滤镜。

## 验证

```bash
bash scripts/video/check-artwork.sh
```

检查六条腿确实相对身体改变姿态、角色平移保留几何、所有策略持续前进、比较组受到相同扰动、固定组不改参数以及轨迹实际不同。合成入口还检查逐幕帧数、最终时长、音轨长度、无字幕轨和完整解码。视觉检查应取连续运动帧，不只看尾帧。

## 说明性模拟的边界

`production/toy_motion.py`是视频专用的小型跟踪模型，不调用或修改研究模型。三条轨迹来自共同的局部控制律和外部扰动，区别是更新策略。选择性更新采用单独合成流上的局部门校准，仅为教学替身；不把其表现称作MathHackson的算法胜利。门的后到信用图仍是拟议机制，不冒充已经完成的正式在线训练。

片中所有英文句子、角色和场景为本项目重新编写；使用ManimGL固定提交`fafa083a4fb274bba9cabde0b6e2f50ba6da0622`。公式使用LaTeX basic模板，英文合成旁白使用Microsoft en-US-AndrewNeural，不克隆任何人的声音。不分发字体、科研数据或模型。

旧的`storyboard-v0.md`、`storyboard-v3-self-learning.md`及`story_prototype.py`仅为历史说明。被否定的第一版实现已从当前提交链回退，原媒体与忽略目录中的源码备份保留，主checkout和V2V不变。
