"""公开模型包不依赖忽略目录，且只含经过核验的参数归档。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("interactive_model_package", ROOT / "scripts/interactive/package-models.py")
assert SPEC is not None and SPEC.loader is not None
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)


def test_manifest_hashes_schema_and_loaders_without_source_logs() -> None:
    manifest = PACKAGE.verify()
    assert len(manifest.models) == 25
    assert sum(m.bytes for m in manifest.models) == 171621
    for kind, count, parameters in (("motor", 1, 82), ("memory", 8, 1429), ("mlp", 8, 1650), ("gate", 8, 141)):
        selected = [m for m in manifest.models if m.kind == kind]
        assert len(selected) == count
        assert all(m.parameter_count == parameters for m in selected)
        assert all(not Path(m.path).is_absolute() and ".." not in Path(m.path).parts for m in selected)
        assert all(m.source.startswith("logs/") for m in selected)
    assert all("不用于现场接受判断" in m.qualification for m in manifest.models if m.kind == "gate")


@pytest.mark.parametrize("kind", ["motor", "memory", "mlp", "gate"])
def test_unknown_archive_content_is_rejected(tmp_path: Path, kind: str) -> None:
    source = next(item for item in PACKAGE.sources() if item.kind == kind)
    tampered = tmp_path / "tampered.npz"
    shutil.copyfile(PACKAGE.PACKAGE / source.path, tampered)
    with zipfile.ZipFile(tampered, "a") as archive:
        archive.writestr("unexpected-samples.npy", b"not a parameter")
    with pytest.raises(ValueError, match="未知或重复"):
        PACKAGE.audit(tampered, kind)
