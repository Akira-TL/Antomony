# v4视频制作入口

此目录隔离新版制作命令，不修改v2/v3入口或媒体。

在工作树根目录执行：

```bash
bash scripts/video/progress/run.sh audio
bash scripts/video/progress/run.sh still all draft
bash scripts/video/progress/run.sh render all hd
bash scripts/video/progress/run.sh assemble
bash scripts/video/progress/run.sh review
```

交互检查：`bash scripts/video/progress/run.sh interactive C03Eligibility draft LINE`。字幕为外置英中双行SRT。只使用v4输出目录。
