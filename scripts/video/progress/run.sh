#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/video-v4 communication/video/rendered/v4
MODE="${1:-render}"
shift "$(( $# > 0 ? 1 : 0 ))"
case "$MODE" in
 audio) exec uv run --no-project --python 3.12 --with 'edge-tts==7.2.8' --with numpy -m communication.video.progress.voice "$@";;
 assemble|check|review|package) exec uv run --no-project --python 3.12 --with numpy -m communication.video.progress.build "$MODE" "$@";;
 render|still|preview|interactive) ;;
 *) echo 'Usage: progress.sh audio|render|still|preview|interactive|assemble|check|review|package' >&2;exit 2;;
esac
SELECT="${1:-all}";QUALITY="${2:-hd}";shift "$(( $#>=2 ? 2 : $# ))"
SCENES=(C00LearningContinues C01MatchedAnts C02StableAndFast C03Eligibility C04ModulatedChange C05FourSteps C06LearnTheRule C07SeparateTheClaims C08ActionDeadline C09KeepLearning)
if [[ "$SELECT" == all ]];then SELECTED=("${SCENES[@]}");else SELECTED=("$SELECT");fi
case "$QUALITY" in hd) RES=1920x1080;FPS=60;;draft) RES=1280x720;FPS=30;;*) echo 'Choose hd or draft' >&2;exit 2;;esac
OUT="$ROOT/communication/video/rendered/v4/clips-$QUALITY";mkdir -p "$OUT"
COMMIT=fafa083a4fb274bba9cabde0b6e2f50ba6da0622
for SCENE in "${SELECTED[@]}";do
 case " ${SCENES[*]} " in *" $SCENE "*) ;;*) echo "Unknown scene $SCENE" >&2;exit 2;;esac
 ARGS=(communication/video/progress/scenes.py "$SCENE" -r "$RES" --fps "$FPS" -c '#0B0E16' --video_dir "$OUT" --log-level WARNING)
 case "$MODE" in render)ARGS+=(-w --quiet);;still)ARGS+=(-w -s --quiet);;preview)ARGS+=(-p);;interactive)ARGS+=(-se "${1:?Line number required}");;esac
 echo "v4: $SCENE $MODE $QUALITY"
 LOG="logs/video-v4/$SCENE-$MODE-$QUALITY.log"
 if [[ "$MODE" == interactive || "$MODE" == preview ]];then
 uvx --python 3.12 --from "git+https://github.com/3b1b/manim.git@$COMMIT" --with av manimgl "${ARGS[@]}"
 elif ! uvx --python 3.12 --from "git+https://github.com/3b1b/manim.git@$COMMIT" --with av manimgl "${ARGS[@]}" > "$LOG" 2>&1;then
 tail -45 "$LOG" >&2;exit 1
 fi
 echo "Ready: $OUT/$SCENE"
done
