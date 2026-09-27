#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <scene.py> <SceneName> [manimgl 参数...]"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIM_COMMIT="fafa083a4fb274bba9cabde0b6e2f50ba6da0622"
SCENE_FILE="$1"
SCENE_NAME="$2"
shift 2

OUTPUT_DIR="$ROOT/communication/video/rendered"
mkdir -p "$OUTPUT_DIR"

exec uvx --from "git+https://github.com/3b1b/manim.git@${MANIM_COMMIT}" --with av   manimgl "$SCENE_FILE" "$SCENE_NAME"   -w -l --video_dir "$OUTPUT_DIR" "$@"
