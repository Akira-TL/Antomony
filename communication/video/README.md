# MathHackson 视频工作流

本目录用于把当前科研主线转成可解释动画。当前阶段采用 3Blue1Brown 的 ManimGL 交互式工作流，而不是先写完整视频再整段渲染。

## 为什么用 ManimGL

3Blue1Brown 当前公开工作流的核心是：把 Scene 跑到指定代码行进入交互式 IPython，在窗口中直接操作已有对象；选中一段代码复制到剪贴板后调用 `checkpoint_paste()`，即可从该 checkpoint 反复试同一段动画。满意后再记录或整段导出。

这里不把 ManimGL 加进主科研依赖，避免污染训练环境。脚本固定使用 3b1b/manim 的指定 Git 提交，通过 `uvx` 启动。

## 直接预览

在项目根目录：

```bash
bash scripts/video-preview.sh communication/video/scenes/story_prototype.py StoryPrototype
```

WSLg 已可用时会直接弹出 ManimGL 预览窗口。

## Grant 风格的交互式编辑

先找到你想停下的位置行号，然后：

```bash
bash scripts/video-preview.sh communication/video/scenes/story_prototype.py StoryPrototype -se 70
```

此时会在对应行附近进入 IPython，同时保留动画窗口。

在编辑器中复制一个以 checkpoint 注释开头的代码块，例如：

```python
# checkpoint: update-question
self.play(...)
self.wait(...)
```

然后在 IPython 输入：

```python
checkpoint_paste()
```

再次修改并复制同一块，再执行同一个命令，会先回到该 checkpoint 的初始状态，再重新播放。

常用变体：

```python
checkpoint_paste(skip=True)    # 不播动画，直接看结果状态
checkpoint_paste(record=True)  # 把这次交互动画写入文件
reload()                       # 修改源文件后重新加载
```

## 导出低清样片

```bash
bash scripts/video-render.sh communication/video/scenes/story_prototype.py StoryPrototype
```

输出目录：

```text
communication/video/rendered/
```

在 WSL 中可直接：

```bash
explorer.exe communication/video/rendered
```

## 写 Scene 的原则

- 一个 Scene 只承担一个清晰的认知推进。
- 一个 checkpoint 对应一个“观众刚刚理解了什么”的最小动画单元。
- 尽量通过同一对象的移动、变形和重新解释推进叙事，不频繁切成 PPT 式新页面。
- 屏幕文字只保留关键词；完整论证交给旁白。
- 颜色语义固定：蓝色表示已有模型/能力，黄色表示候选修改，绿色表示有利后果，红色表示伤害或失败，灰色表示冻结/对照。
- 正式研究尚未得到的结论不得画成已经成功；当前视频首先表达问题、机制和判别实验。
