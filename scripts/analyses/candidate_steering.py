"""重建已登记四步轨迹的实际方向采样并报告动作变化。"""
import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.comparison.steering_audit import audit


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    input_directory: Path
    manifest: Path
    output: Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    config = Config.model_validate_json(parser.parse_args().config.read_bytes())
    if config.output.exists():
        raise FileExistsError(config.output)
    torch.set_num_threads(1)
    result = audit(config.input_directory, config.manifest)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open('x') as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(exclude={'changes'}, indent=2))
