#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SOURCE="$ROOT/communication/video/rendered/v2"
FILM=Learning_While_Acting_v2_1080p60.mp4
for F in "$FILM" index.html render-manifest.json Learning_While_Acting.en.srt;do test -f "$SOURCE/$F";done
PROFILE="$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')"
DOWNLOADS="$(wslpath -u "$PROFILE")/Downloads"
test -d "$DOWNLOADS"
DEST="$DOWNLOADS/MathHackson_Learning_While_Acting_v2"
if [[ -e "$DEST" ]];then DEST="${DEST}_$(date +%Y%m%d_%H%M%S)";fi
mkdir -p "$DEST" logs/video-v2
for F in "$FILM" index.html render-manifest.json Learning_While_Acting.en.srt toy-provenance.json speech-visual-timeline.json ManimGL_sources_v2.zip;do
 if [[ -f "$SOURCE/$F" ]];then cp -- "$SOURCE/$F" "$DEST/$F";fi
done
printf '%s\n' "$DEST" > logs/video-v2/windows-review-path.txt
WIN="$(wslpath -w "$DEST/index.html")"
ESCAPED="${WIN//\'/\'\'}"
powershell.exe -NoProfile -NonInteractive -Command "Start-Process -FilePath '$ESCAPED'" > logs/video-v2/open-review.log 2>&1
printf 'Review: %s\nVideo: %s\n' "$WIN" "$(wslpath -w "$DEST/$FILM")"
