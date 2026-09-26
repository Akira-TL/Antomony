"""已查看标签后的探索性候选覆盖核验，不拟合模型。"""
import argparse
from pathlib import Path

import torch

from mathhackson.training.comparison.candidate_label_coverage import Config, run


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
result = run(Config.model_validate_json(args.config.read_text()))
print(result.model_dump_json(include={'source_commit', 'started_at', 'completed_at', 'elapsed_seconds', 'audit', 'partitions'}, indent=2))
