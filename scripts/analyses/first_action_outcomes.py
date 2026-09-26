"""独立运行首步诊断，不提供训练或部署接口。"""
import argparse
import hashlib
from pathlib import Path

from mathhackson.training.comparison.first_action import Plan, run
from mathhackson.training.comparison.first_action_audit import audit


parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('sample', 'audit'))
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
plan = Plan.model_validate_json(args.config.read_text())
raw = Path('logs/first-action-outcomes/A001')
if args.phase == 'sample':
    result = run(plan, raw)
    manifest = ''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(raw)}\n'
                       for path in sorted(raw.rglob('*')) if path.is_file())
    Path('data/first-action-outcomes/manifest.sha256').write_text(manifest)
    print(result.model_dump_json(indent=2))
else:
    from mathhackson.training.comparison.auditing import verify_manifest
    verify_manifest(raw, Path('data/first-action-outcomes/manifest.sha256'))
    result = audit(raw)
    output = Path('.research/analysis/first-action-outcomes/A001/outputs/summary.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(indent=2))
