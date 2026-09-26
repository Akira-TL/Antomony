"""固定既有输入身份后重建更新约束，不采样新世界。"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from mathhackson.training.comparison.update_constraints import audit


class Identity(BaseModel):
    path: Path
    sha256: str


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    input_directory: Path
    manifest: Path
    identities: list[Identity]
    output: Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    config = Config.model_validate_json(parser.parse_args().config.read_bytes())
    if config.output.exists():
        raise FileExistsError(config.output)
    for identity in config.identities:
        if hashlib.sha256(identity.path.read_bytes()).hexdigest() != identity.sha256:
            raise ValueError(f'已固定输入身份改变: {identity.path}')
    torch.set_num_threads(1)
    print(datetime.now(timezone.utc).isoformat(), 'start', flush=True)
    print('Python', platform.python_version(), 'numpy', np.__version__, 'torch', torch.__version__, flush=True)
    result = audit(config.input_directory, config.manifest)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with config.output.open('x') as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(exclude={'changes'}, indent=2), flush=True)
    print(datetime.now(timezone.utc).isoformat(), 'complete', flush=True)
