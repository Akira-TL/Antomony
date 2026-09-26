"""从已冻结轨迹核验四步与十六步窗口的生存比较。"""
import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mathhackson.training.comparison.cadence_audit import audit


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
    result = audit(config.input_directory, config.manifest)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open('x') as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(exclude={'sixteen', 'four', 'activity'}, indent=2))
