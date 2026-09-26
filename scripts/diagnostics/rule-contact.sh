#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
uv run --no-sync --offline python scripts/diagnostics/rule_contact.py \
  2>&1 | tee "logs/rule-contact-$RUN_ID.jsonl"
