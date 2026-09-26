#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODE="${1:-render}"
SELECT="${2:-all}"
QUALITY="${3:-hd}"
shift "$(( $#>=3 ? 3 : $# ))"
COMMIT=fafa083a4fb274bba9cabde0b6e2f50ba6da0622
SCENES=(A01FollowTheTrail A02SameAntDifferentLearning A03WeightBecomesMovement A04LearnFromTheGap A05NoiseOrChange A06LearningTheChoice A07UndoOneChange A08KeepWalking A09ReturnToTheAnts)
case "$QUALITY" in hd) RES=1920x1080;FPS=60;; draft) RES=1280x720;FPS=30;; *) echo 'quality must be hd or draft' >&2;exit 2;; esac
OUT="$ROOT/communication/video/rendered/v2/clips-$QUALITY"
mkdir -p "$OUT" logs/video-v2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$SELECT" == all ]]; then SELECTED=("${SCENES[@]}"); else SELECTED=("$SELECT"); fi
for SCENE in "${SELECTED[@]}"; do
 case " ${SCENES[*]} " in *" $SCENE "*) ;; *) echo "unknown scene $SCENE" >&2;exit 2;;esac
 ARGS=(communication/video/scenes/ants_learning.py "$SCENE" -r "$RES" --fps "$FPS" -c '#0B0E16' --video_dir "$OUT" --log-level WARNING)
 case "$MODE" in render) ARGS+=(-w --quiet);; still) ARGS+=(-w -s --quiet);; preview) ARGS+=(-p);; interactive) ARGS+=(-se "${1:?line number required}");; *) echo 'render|still|preview|interactive required' >&2;exit 2;; esac
 echo "Rendering $SCENE ($MODE $RES $FPS fps)"
 LOG="logs/video-v2/$SCENE-$MODE-$QUALITY.log"
 if [[ "$MODE" == interactive || "$MODE" == preview ]];then
 uvx --python 3.12 --from "git+https://github.com/3b1b/manim.git@$COMMIT" --with av manimgl "${ARGS[@]}"
 elif ! uvx --python 3.12 --from "git+https://github.com/3b1b/manim.git@$COMMIT" --with av manimgl "${ARGS[@]}" > "$LOG" 2>&1;then
 tail -45 "$LOG" >&2;exit 1
 fi
 echo "Ready: $OUT/$SCENE"
done
