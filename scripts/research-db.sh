#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="$ROOT/.agents/skills/akira-research/scripts/research_db.py"
if [[ ! -f "$OWNER" ]]; then
    printf '%s\n' '缺少项目科研技能引用，请先运行 bash scripts/setup-skills.sh' >&2
    exit 1
fi
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec uv run --no-project "$OWNER" --project "$ROOT" "$@"
