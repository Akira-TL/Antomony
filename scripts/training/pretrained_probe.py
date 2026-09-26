"""离线验收已下载的官方 μLO 权重；不是蚁群学习效果实验。"""
from __future__ import annotations

import copy
import hashlib
import os
import sys
import time
from pathlib import Path

import torch
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "logs/external-models/pylo"
WEIGHTS = ROOT / "logs/external-models/mulo"
MODEL_HASH = "ad7c499d1bd389c768f20019a6e1a423c51c7d1db64774d038f4fe56bf21ccb3"


class ProbeResult(BaseModel):
    purpose: str = "预训练优化器 CPU 接入验收，不是蚁群自训练效果"
    code_commit: str = "d05c6c1e4f4d7c46a91828a0826c2699bfe48211"
    checkpoint_sha256: str = MODEL_HASH
    steps: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float
    finite: bool
    optimizer_weights_unchanged: bool
    restored_next_step_equal: bool
    torch_version: str
    torch_path: str


def main() -> None:
    if hashlib.sha256((WEIGHTS / "model.safetensors").read_bytes()).hexdigest() != MODEL_HASH:
        raise ValueError("外部权重哈希不一致，拒绝加载")
    if hashlib.sha256((WEIGHTS / "config.json").read_bytes()).hexdigest() != (
        "03ce572db3d332984fe31817c270d486a38ee26f68277b2f9911ef8284433b94"
    ):
        raise ValueError("外部权重配置哈希不一致，拒绝加载")
    for relative, expected in (
        ("pylo/optim/AdafacLO_naive.py", "2b2e7b69f38b27f18c383e7d26fb2b67e9a0456c7552556c21668a3b1dc0636d"),
        ("pylo/optim/MuLO_naive.py", "cbd8560e5cf33a3acb08c69d8b349c38b6ebbe01d7d1d18d2ed131e266d1c7b1"),
        ("pylo/models/Meta_MLP.py", "9d8daded907b9b3d793762545e42a58d895a8819bc356d32a0ea1da375939ba0"),
    ):
        if hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"外部核心源码 {relative} 哈希不一致，拒绝加载")
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(SOURCE))
    import mup
    from pylo.optim import MuLO_naive

    torch.set_num_threads(1)
    torch.manual_seed(72)
    model = torch.nn.Linear(2, 2)
    mup.set_base_shapes(model, model)
    optimizer = MuLO_naive(model.parameters(), lr=1., hf_key=str(WEIGHTS))
    optimizer.network.requires_grad_(False)
    original = [p.detach().clone() for p in optimizer.network.parameters()]
    x = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    y = torch.tensor([[.7, -.3], [.2, .9], [-.7, .3], [-.2, -.9]])

    def advance(network: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
        opt.zero_grad()
        loss = (network(x) - y).square().mean()
        loss.backward()
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in network.parameters()):
            raise ValueError("预训练优化器产生非有限参数")

    initial = float((model(x) - y).square().mean().detach())
    start = time.perf_counter()
    for _ in range(128):
        advance(model, optimizer)
    elapsed = time.perf_counter() - start
    final = float((model(x) - y).square().mean().detach())
    restored = copy.deepcopy(model)
    mup.set_base_shapes(restored, restored, rescale_params=False)
    restored_optimizer = MuLO_naive(restored.parameters(), lr=1., hf_key=str(WEIGHTS))
    restored_optimizer.network.requires_grad_(False)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    advance(model, optimizer)
    advance(restored, restored_optimizer)
    result = ProbeResult(
        steps=128, initial_loss=initial, final_loss=final, elapsed_seconds=elapsed, finite=True,
        optimizer_weights_unchanged=all(
            a.equal(b) for a, b in zip(original, optimizer.network.parameters(), strict=True)),
        restored_next_step_equal=all(
            a.equal(b) for a, b in zip(model.parameters(), restored.parameters(), strict=True)),
        torch_version=torch.__version__, torch_path=torch.__file__,
    )
    path = ROOT / "logs/external-models/mulo/cpu-probe.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(indent=2))
    if not (final < initial and result.optimizer_weights_unchanged and result.restored_next_step_equal):
        raise SystemExit("CPU 接入验收未通过，保留结果，不宣称可用于仿真")


if __name__ == "__main__":
    main()
