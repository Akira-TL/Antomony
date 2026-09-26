"""核验并按字节复制实时验收所需的固定模型，不运行训练。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal
import zipfile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mathhackson.training.direction.checkpoint import MotorSnapshot, load_motor
from mathhackson.training.foraging.memory import MemoryPolicy
from mathhackson.training.foraging.mlp import FeedforwardPolicy
from mathhackson.training.foraging.update_decision import UpdateDecision


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/interactive"
Kind = Literal["memory", "mlp", "gate", "motor"]


class ArrayRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    shape: tuple[int, ...]
    dtype: str
    parameter: bool


class ModelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Kind
    initialization: int | None
    path: str
    source: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    parameter_count: int = Field(gt=0)
    qualification: str
    arrays: tuple[ArrayRecord, ...]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["interactive-model-package-v1"] = "interactive-model-package-v1"
    qualification: str = "工程验收初始化；未证明自训练优势或学会更新时机"
    audit: str = "逐个核验归档键、类型、维度及白名单元数据；不包含对象数组、样本或凭据字段"
    models: tuple[ModelRecord, ...]


@dataclass(frozen=True)
class Source:
    kind: Kind
    initialization: int | None
    path: str
    source: str
    qualification: str


def sources() -> tuple[Source, ...]:
    result = [Source("motor", None, "motor/update-001200.npz",
        "logs/direction-motor/20260926T053106-2/seed-41/update-001200.npz",
        "纯方向输入动作模型；现场冻结，不提供目标位置或距离")]
    for i in range(8):
        result.extend((
            Source("memory", i, f"memory/episode-0000-ant-{i:02d}.npz",
                f"logs/memory-check/20260926T112254-2/episode-0000-ant-{i:02d}.npz",
                "14宽基础模型加零初始化记忆连接；记忆冻结，未证明记忆收益"),
            Source("mlp", i, f"mlp/seed-{81 + i}-signal-002400.npz",
                f"logs/matched-foundation/20260926T121232-2/seed-{81 + i}-signal-002400.npz",
                "17宽普通MLP基础模型；现场冻结，不使用循环连接"),
            Source("gate", i, f"gate/ant-{i:02d}/step-0200.npz",
                f"logs/model-artifacts/memory-acceptance/ant-{i:02d}/step-0200.npz",
                "旧目标200步接受模型；未通过采用条件；仅兼容当前加载和保存接口，不用于现场接受判断"),
        ))
    return tuple(result)


def audit(path: Path, kind: Kind) -> tuple[int, tuple[ArrayRecord, ...]]:
    if kind == "motor":
        model, metadata = load_motor(path)
        expected = {"encoder_weight": (16, 2), "encoder_bias": (16,),
                    "readout_weight": (2, 16), "readout_bias": (2,)}
        meta_names = {"metadata"}
        if metadata.seed != 41 or metadata.updates != 1200:
            raise ValueError("动作模型来源身份错误")
    elif kind == "memory":
        model = MemoryPolicy.load(path)
        if any(bool(p.any()) for p in model.memory_parameters()):
            raise ValueError("预期零初始化记忆连接")
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        meta_names = {"version", "update", "phase", "recent_lags", "sparse_lags"}
    elif kind == "mlp":
        model = FeedforwardPolicy.load(path)
        model.assert_reserved()
        if model.hidden_width != 17:
            raise ValueError("预期17宽普通MLP")
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        meta_names = {"version", "update", "phase", "hidden_width"}
    else:
        model = UpdateDecision.load(path)
        if model.profile != "parameters" or model.updates != 200:
            raise ValueError("预期旧目标200步接受模型")
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        meta_names = {"version", "updates"}
    allowed = set(expected) | meta_names
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(allowed) or set(names) != {f"{key}.npy" for key in allowed} or archive.comment:
            raise ValueError("归档包含未知或重复内容")
    arrays = []
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(allowed):
            value = archive[key]
            if key in expected:
                if value.dtype != np.dtype("float32") or value.shape != expected[key] or not np.isfinite(value).all():
                    raise ValueError(f"参数字段无效：{key}")
            elif key in ("update", "updates", "hidden_width"):
                wanted = 200 if key == "updates" else 17 if key == "hidden_width" else 0 if kind == "memory" else 2400
                if value.shape != () or value.dtype.kind not in "iu" or int(value) != wanted:
                    raise ValueError(f"训练元数据无效：{key}")
            elif key in ("recent_lags", "sparse_lags"):
                wanted_lags = (1, 2, 3, 4) if key == "recent_lags" else (4, 8, 12, 16)
                if value.dtype.kind not in "iu" or value.shape != (4,) or tuple(value) != wanted_lags:
                    raise ValueError("记忆步长元数据无效")
            elif key == "metadata":
                if value.shape != () or value.dtype.kind != "U":
                    raise ValueError("动作元数据须为JSON字符串标量")
                MotorSnapshot.model_validate_json(str(value.item()))
            else:
                wanted_text = ({"memory": "local-mlp-memory-v1", "mlp": "local-mlp-v2", "gate": "local-update-decision-v1"}[kind]
                               if key == "version" else "frozen" if kind == "memory" else "signal")
                if value.shape != () or value.dtype.kind != "U" or str(value.item()) != wanted_text:
                    raise ValueError(f"文本元数据无效：{key}")
            arrays.append(ArrayRecord(name=key, shape=value.shape, dtype=str(value.dtype), parameter=key in expected))
    return sum(p.numel() for p in model.parameters()), tuple(arrays)


def record(source: Source, path: Path) -> ModelRecord:
    count, arrays = audit(path, source.kind)
    content = path.read_bytes()
    return ModelRecord(kind=source.kind, initialization=source.initialization, path=source.path,
        source=source.source, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
        parameter_count=count, qualification=source.qualification, arrays=arrays)


def build() -> Manifest:
    records = tuple(record(source, ROOT / source.source) for source in sources())
    manifest = Manifest(models=records)
    for item in records:
        target = PACKAGE / item.path
        if target.exists():
            raise FileExistsError(f"拒绝覆盖：{target}")
    if (PACKAGE / "manifest.json").exists():
        raise FileExistsError("拒绝覆盖已有清单")
    for item in records:
        target = PACKAGE / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (ROOT / item.source).read_bytes()
        if hashlib.sha256(content).hexdigest() != item.sha256:
            raise ValueError("核验后源文件改变")
        with target.open("xb") as stream:
            stream.write(content)
    with (PACKAGE / "manifest.json").open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    return verify(check_sources=True)


def verify(*, check_sources: bool = False) -> Manifest:
    manifest = Manifest.model_validate_json((PACKAGE / "manifest.json").read_text())
    expected = sources()
    if len(manifest.models) != len(expected):
        raise ValueError("模型清单数量错误")
    for source, saved in zip(expected, manifest.models, strict=True):
        current = record(source, PACKAGE / source.path)
        if saved != current:
            raise ValueError(f"模型内容或清单不匹配：{source.path}")
        if check_sources and (ROOT / source.source).read_bytes() != (PACKAGE / source.path).read_bytes():
            raise ValueError(f"原字节复制不一致：{source.path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="只在原始模型所在机器首次打包")
    parser.add_argument("--source-check", action="store_true", help="同时与原始logs文件逐字节比较")
    args = parser.parse_args()
    result = build() if args.build else verify(check_sources=args.source_check)
    print(f"模型包核验通过：{len(result.models)}个文件，{sum(m.bytes for m in result.models)}字节")
