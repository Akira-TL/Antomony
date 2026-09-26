"""核对固定训练与配对基础轨迹，再生成描述结果。"""
import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mathhackson.training.comparison.matched_audit import audit


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_directory: Path
    manifest: Path
    protocol: Path
    output: Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = Config.model_validate_json(parser.parse_args().config.read_text())
    result = audit(config.input_directory, config.manifest, config.protocol)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open("x") as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(exclude={"worlds"}, indent=2))
