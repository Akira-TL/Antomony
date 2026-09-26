"""核验固定数据并描述两种动作归因范围的配对差，不训练模型。"""
import argparse
from pathlib import Path

import torch

from mathhackson.training.comparison.credit_audit import Config, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(run(Config.model_validate_json(args.config.read_text())).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
