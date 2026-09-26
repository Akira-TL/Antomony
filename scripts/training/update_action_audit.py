"""逐项重建既有候选写入，测量同一观察上的动作分布变化。"""
from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np
from pydantic import BaseModel
import torch

from novel_adaptation_probe import Artifact, Frame, ProbeConfig
from mathhackson.training.direction.checkpoint import load_motor
from mathhackson.training.foraging.policy import ForagingPolicy
from mathhackson.training.foraging.reward import direction_distribution


class AuditConfig(BaseModel):
    source_commit: str
    input_directory: str
    inputs: list[Artifact]


class Change(BaseModel):
    condition: str
    mode: str
    seed: int
    generation: int
    tick: int
    ant: int
    accepted: bool
    changed: bool
    negative_feedback: bool
    novel_response: float
    gradient_norm: float
    actual_delta_norm: float
    direction_change: float
    probability_total_variation: float
    move_changed: bool
    turn_change: float
    away_alignment_change: float | None


def set_weights(model: ForagingPolicy, value: np.ndarray) -> None:
    with torch.no_grad():
        model.novel_signal.weight.copy_(torch.from_numpy(value[:40].reshape(8, 5)))
        model.novel_strength.weight.copy_(torch.from_numpy(value[40:].reshape(8, 5)))


def run(config: ProbeConfig, root: Path, output: Path, condition: str, seed: int, mode: str) -> None:
    directory = root / f"{condition}-{seed}-{mode}"
    models = [ForagingPolicy.load(directory / f"generation-00-ant-{i:02d}.npz") for i in range(config.environment.ants)]
    stable = [torch.cat([p.detach().flatten() for p in model.reserved_parameters()]).numpy().copy() for model in models]
    fast = [np.zeros(80, dtype=np.float32) for _ in models]
    motor, _ = load_motor(Path(config.sources[-1].path))
    for generation in range(1, config.generations + 1):
        histories: list[deque[torch.Tensor]] = [deque(maxlen=16) for _ in models]
        path = directory / f"generation-{generation:02d}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream, (output / "changes.jsonl").open("a", encoding="utf-8") as result, torch.inference_mode():
            for line in stream:
                frame = Frame.model_validate_json(line)
                for ant in frame.ants:
                    if not ant.active:
                        continue
                    i, model = ant.index, models[ant.index]
                    observation = torch.tensor(ant.observation)
                    history = tuple(histories[i])
                    before, hidden, _ = model(observation, history)
                    if not torch.allclose(hidden, torch.tensor(ant.hidden), rtol=0., atol=1e-5):
                        raise AssertionError("轨迹隐藏状态不能按原权重重建")
                    item = ant.proposal
                    if item is not None:
                        delta = np.asarray(item.delta, dtype=np.float32).copy()
                        if not item.accepted:
                            delta.fill(0.)
                        while float(np.linalg.norm(fast[i] + delta)) > config.adaptation.maximum_residual_norm and np.any(delta):
                            delta *= .5
                        previous = stable[i] + fast[i]
                        changed = bool(np.any(delta) and np.any(previous + delta != previous))
                        if changed != item.changed:
                            raise AssertionError("参数写入标记不一致")
                        if changed:
                            fast[i] += delta
                            set_weights(model, stable[i] + fast[i])
                        after, _, _ = model(observation, history)
                        cross = float(before[0] * after[1] - before[1] * after[0])
                        dot = float(before @ after)
                        angle = abs(math.degrees(math.atan2(cross, dot)))
                        variation = float((direction_distribution(after).probs - direction_distribution(before).probs).abs().sum() / 2.)
                        first, second = motor.decide(before.numpy()), motor.decide(after.numpy())
                        novel = torch.expm1(observation[:72].reshape(9, 8) * math.log(9.))[:, 3:].sum(-1)
                        away = -torch.stack((novel[1] - novel[3], novel[2] - novel[4]))
                        strength = float(away.norm())
                        alignment = float((after - before) @ (away / strength)) if strength > .0001 else None
                        measured = Change(condition=condition, mode=mode, seed=seed, generation=generation,
                                          tick=frame.tick, ant=i, accepted=item.accepted, changed=changed,
                                          negative_feedback=item.minimum_reward < 0., novel_response=item.novel_response,
                                          gradient_norm=item.gradient_norm, actual_delta_norm=float(np.linalg.norm(delta)),
                                          direction_change=angle, probability_total_variation=variation,
                                          move_changed=first.move != second.move, turn_change=abs(first.turn - second.turn),
                                          away_alignment_change=alignment)
                        result.write(measured.model_dump_json() + "\n")
                    histories[i].append(torch.tensor(ant.hidden))
        for i, model in enumerate(models):
            saved = ForagingPolicy.load(directory / f"generation-{generation:02d}-ant-{i:02d}.npz")
            if any(not value.equal(saved.state_dict()[key]) for key, value in model.state_dict().items()):
                raise AssertionError("按提案重建的代末模型不同于实际快照")
            with np.load(directory / f"generation-{generation:02d}-ant-{i:02d}.residual.npz", allow_pickle=False) as checkpoint:
                np.testing.assert_array_equal(fast[i], checkpoint["fast"])
        print(f"verified {condition} {seed} {mode} generation={generation}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("logs/novel-adaptation/20260926T072832-2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = ProbeConfig.model_validate_json((args.input / "config.json").read_text())
    if config.profile == "direction":
        raise ValueError("本审计只重建80维信号模块，不适用于45维方向修正")
    paths = sorted(path for path in args.input.rglob("*") if path.is_file())
    audit = AuditConfig(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        input_directory=str(args.input),
                        inputs=[Artifact(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths])
    (args.output / "config.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
    for condition in config.conditions:
        for seed in config.seeds:
            for mode in config.modes:
                run(config, args.input, args.output, condition, seed, mode)
    if any(hashlib.sha256(Path(item.path).read_bytes()).hexdigest() != item.sha256 for item in audit.inputs):
        raise AssertionError("重建审计改变了输入文件")


if __name__ == "__main__":
    main()
