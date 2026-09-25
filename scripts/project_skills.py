"""建立或检查项目级技能引用；不更新机器级技能源。"""
from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ['name', 'repository', 'ref', 'commit', 'source_path']


@dataclass(frozen=True)
class Skill:
    name: str
    repository: str
    ref: str
    commit: str
    source_path: str


def load_skills(path: Path) -> tuple[Skill, ...]:
    with path.open(encoding='utf-8', newline='') as handle:
        rows = csv.reader(handle, delimiter='\t')
        if next(rows, None) != HEADER:
            raise ValueError('技能清单表头不符合约定')
        skills = tuple(Skill(*row) for row in rows if row)
    if len({item.name for item in skills}) != len(skills):
        raise ValueError('技能清单存在重复名称')
    for item in skills:
        if not item.name or Path(item.name).name != item.name or item.name in {'.', '..'}:
            raise ValueError(f'非法技能名称：{item.name}')
    return skills


def git_value(path: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def validate_source(skill: Skill, registry: Path) -> Path:
    source = registry / skill.name
    if not source.is_symlink() or not (source / 'SKILL.md').is_file():
        raise RuntimeError(f'机器级技能缺失或不是受管引用：{source}')
    resolved = source.resolve(strict=True)
    repository = Path(git_value(resolved, 'rev-parse', '--show-toplevel'))
    origin = git_value(repository, 'remote', 'get-url', 'origin')
    if origin.removesuffix('.git') != skill.repository.removesuffix('.git'):
        raise RuntimeError(f'{skill.name} 来源不同：{origin}')
    if resolved != repository / skill.source_path:
        raise RuntimeError(f'{skill.name} 来源路径发生变化')
    current = git_value(repository, 'rev-parse', 'HEAD')
    if current != skill.commit:
        raise RuntimeError(f'{skill.name} 来源版本已变化：{current}，请显式复核后更新项目清单')
    return source


def setup(*, check_only: bool = False) -> int:
    skills = load_skills(ROOT / '.agents/skill-view.tsv')
    registry = Path.home() / '.agents/skills'
    view = ROOT / '.agents/skills'
    targets = tuple((item, validate_source(item, registry)) for item in skills)
    for item, source in targets:
        link = view / item.name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.readlink() != source:
                raise RuntimeError(f'项目内已有不同内容，拒绝覆盖：{link}')
        elif check_only:
            raise RuntimeError(f'缺少项目引用：{link}')
    if not check_only:
        view.mkdir(parents=True, exist_ok=True)
        for item, source in targets:
            link = view / item.name
            if not link.is_symlink():
                link.symlink_to(source, target_is_directory=True)
    print(f'项目级技能引用及来源版本检查通过：{len(skills)} 项')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='只检查，不建立引用')
    args = parser.parse_args()
    try:
        return setup(check_only=args.check)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'错误：{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
