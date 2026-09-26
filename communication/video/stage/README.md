# Learning While Acting｜v3路演版

## 当前交付

180秒，1920×1080，60fps；新增30秒提问式引言，缩短重复讲解，末尾从蚂蚁回到架构问题。英文合成旁白；无烧录字幕、无内嵌字幕轨。SRT独立提供，审片页不自动加载。

成片：`../rendered/v3/Learning_While_Acting_v3_Stage_1080p60.mp4`。

v2的源码、旁白、时间表和原片不修改。A04、A05、A06、A08合计92秒的已认可视频及分段旁白逐字节复制。其余88秒新渲染，不统一倍速处理旧片。

## 操作

在MathHackson工作树根目录执行：

```bash
bash scripts/video/stage.sh audio
bash scripts/video/stage.sh render all hd
bash scripts/video/stage.sh assemble
bash scripts/video/stage.sh review
```

新输出只进入`communication/video/rendered/v3/`，日志只进入`logs/video-v3/`。最后一个命令会校验成片、生成源码包、复制到独立的Windows v3下载文件夹并打开审片页；目录已经存在时加时间戳，不覆盖旧版。

本剪辑依赖已生成的v2四段`normalized-hd/`、原分段`audio/*-timed.wav`、`speech-visual-timeline.json`和视频教学用的`toy-gate.npy`。缺失时应先用原视频命令重建v2，不从研究项目提取其他模型或用新训练结果替代。

单幕预览、低清和检查点：

```bash
bash scripts/video/stage.sh render B00TheQuestion draft
bash scripts/video/stage.sh preview B00TheQuestion draft
grep -n 'checkpoint:' communication/video/stage/scenes.py
bash scripts/video/stage.sh interactive B00TheQuestion draft LINE_NUMBER
```

复制以检查点注释开头的片段后，在ManimGL终端调用`checkpoint_paste()`重播；`checkpoint_paste(record=True)`录制局部动画。整片录音由独立合成入口放置，不依赖交互终端。

## 数学与画面

引言的小网络由确定的矩阵计算产生活动：训练示意的参数在1.25秒后固定，而输入和节点活动继续变化。预测与延迟到达的观测分开；环境变化后出现的差值引出局部修改问题，不提前画成已修复。局部候选用黄色虚线，不永久否定反向传播。

引言最后两秒与下一幕共享同一个教学世界、坐标系和连续模拟时钟；切点对应同一位置、朝向与步相。蚂蚁继续行走。缩短双蚂蚁段保留原教学模型速度，只不再播放后面的重复等待。参数段仅保留一次完整滑动。撤回只减去一笔仍可归属的变化，旧轨迹不倒放。

所有新画面与旁白均为本项目重新编写。参考3Blue1Brown的活动/权重区分、参数刻度、提问后进入具体实例的表达方法；没有复制原片、场景代码、角色、声纹或第三方图像。ManimGL固定版本仍为`fafa083a4fb274bba9cabde0b6e2f50ba6da0622`。

## 实际检查记录

- 六个新Scene均执行：引言与接入蚂蚁的44秒先做720p草稿，其余四幕先检查末状态，再完整1080p60渲染。
- 合成器逐幕检查帧数、分辨率、帧率，去掉可能的额外终止帧；总帧数10800、音频8640000采样，均为180秒。整片完整解码通过。
- 四段核心视频和原旁白用SHA-256检查逐字节复用；v2主视频仍为`4e7d70eaf9a4336972ff84a09a01d6a132ef57692958e24ff324708fcb61bedf`。十个指定v2源码文件与规划基线提交`77ec519`逐字节相同。
- 最终MP4只有video和audio流；合成链只映射这两个输入，不使用字幕滤镜或字幕输入。
- 从最终文件抽取16个画面，覆盖引言各阶段、29.98/30.00秒接片、各主体段和结尾。接片前后图像平均绝对像素差约0.317/255；它只检查是否突跳，不代替人类理解评价。
- 新配音按TTS句界与视觉时点放置完整句子；保留原MP3和元数据。一条请求暂时无音频，有限重试后成功。没有用截断句子或整体视频加速隐藏问题。
- 未宣称真人听感测试、独立观众验收或10800帧全部逐帧人工检查。

## 非隔离双轴复查

起点`77ec519`，需求依据`../plans/stage-cut-v3.md`与用户本轮执行授权。当前会话先检查规范、再检查分镜，不冒充独立审阅者。

规范轴：新增文件全部位于独立stage目录和scripts/video入口；旧v2文件及主研究未修改；uv隔离依赖、日志与媒体排除Git、Guard提交与脚本/Python语法检查通过。隔离环境中的ManimGL实际渲染通过，不将主环境语言服务器缺依赖警告描述为完整类型检查通过。

分镜轴：30秒引言、连续接入、92秒核心原片、缩短重复段、架构回扣、全英文、独立字幕与180秒均由源码/媒体检查支持。没有增加研究能力、流程框或虚构比赛成绩。教学模型及局部信用仍保留示意/提案身份。

最终SHA-256、字节数、来源提交与逐段复用信息见`../rendered/v3/render-manifest.json`。
