# 预训练参数更新器的最小接入调查

核验日期：2026-09-26。调查阶段核验官方源码、作者权重目录及模型卡；随后主执行者已下载固定提交源码和 μLO 权重，并核对内容哈希。依赖隔离和恢复流程修复后，CPU 简单损失更新及状态恢复检查已通过。现有环境仍是 `torch 2.8.0+cpu`，没有 JAX。本文不是蚁群迁移效果证据，也不替代正式研究设计。

## 结论

优先尝试作者实验室 [PyLO](https://github.com/Belilovsky-Lab/pylo) 的 `MuLO_naive` 及其 `btherien/mulo` 预训练权重，理由是权重只有 10,096 字节，有纯 PyTorch 实现，避免引入完整 JAX 训练栈。先做短小、隔离的 CPU 加载与更新检查；失败时不要无限修补外部库。PyLO 的 VeLO 可作第二候选，原版 JAX VeLO 与 Celo 暂不优先。

这些权重学习的是“给定损失梯度后如何更新参数”，不是已训练好的蚂蚁，也不会替项目定义正确的在线损失。已检查的实现均没有显式的学习型离散跳过门控；更新很小不能自动算“学会不更新”。我们仍须定义只利用已发生观察的损失或梯度估计，再检验更新是否改善后续行为。

## 候选比较

| 候选 | 权重已发布 | 输入和更新特征 | CPU 接入判断 | 许可 |
| --- | --- | --- | --- | --- |
| Google VeLO 原版 | 官方 `pretrained_optimizers.py` 指向 Google Cloud Storage，见下文；作者 Celo 仓库另提供格式转换版 | 梯度、损失历史、参数及统计量、训练总步数；循环调节更新规则 | 原生 JAX，不能直接作为 PyTorch 优化器使用；本项目还需桥接或改写，成本较高 | Google 代码 Apache-2.0 |
| Celo | `amoudgl/celo/theta.state`，149,216 字节 | 梯度、损失、参数统计、总步数；小型更新网络及循环调度器 | 原生 JAX；当前官方依赖声明含 Python >=3.12、`jax[cuda12]`、TensorFlow 相关包，不宜直接同步进 CPU 项目 | 仓库 MIT，部分继承代码 Apache-2.0；模型卡 MIT |
| PyLO 的 μLO | `btherien/mulo/model.safetensors`，10,096 字节 | 每参数 39 维特征，包括参数、梯度、动量、二阶统计、步数；输出连续更新方向和幅度，`step(loss=None)` 不使用损失值 | `MuLO_naive` 使用纯 PyTorch；须设置 μP 参数形状信息。当前源码按模型参数设备选择 CPU/GPU，未实际运行验证 PyTorch 2.8 | 模型卡 Apache-2.0；PyLO 根 LICENSE 为 BSD-3-Clause，但 setup.py 标为 Apache-2.0，须保留原始许可而非混写 |
| PyLO 的 VeLO | RNN 9,119,908 字节，MLP 164,296 字节 | `step(loss)` 与参数梯度，初始化指定 `num_steps`；更大的循环更新器 | 同仓库 `VeLO_naive`，支持按参数设备放置状态；比 μLO 开销大，CPU 用时未知 | 两个权重模型卡均 Apache-2.0；源码许可边界同上 |

来源：[VeLO 预训练注册表](https://github.com/google/learned_optimization/blob/main/learned_optimization/research/general_lopt/pretrained_optimizers.py)、[VeLO 封装](https://github.com/google/learned_optimization/blob/main/learned_optimization/research/general_lopt/prefab.py)、[Celo 官方说明](https://github.com/amoudgl/celo)、[Celo 依赖](https://github.com/amoudgl/celo/blob/main/pyproject.toml)、[Celo 更新器](https://github.com/amoudgl/celo/blob/main/celo/optimizers/celo.py)、[μLO 模型卡](https://huggingface.co/btherien/mulo)、[PyLO VeLO 模型卡](https://huggingface.co/Pauljanson002/VeLO_RNN)。

## 可追溯下载位置

PyLO 本次通过 GitHub API 实查的代码版本：`d05c6c1e4f4d7c46a91828a0826c2699bfe48211`。后续下载源码应固定这个提交，不直接执行网上安装脚本。

- μLO 权重：[model.safetensors](https://huggingface.co/btherien/mulo/resolve/main/model.safetensors)，10,096 字节；模型目录同时需要 [config.json](https://huggingface.co/btherien/mulo/resolve/main/config.json)，65 字节。
- μLO 权重目录 API 返回的 LFS SHA256 为 `ad7c499d1bd389c768f20019a6e1a423c51c7d1db64774d038f4fe56bf21ccb3`，下载后须核对。HF 提交版本查询连续 SSL 失败，当前未核实固定提交号；上述 `main` 地址不是版本锁定，只能结合内容哈希保证权重身份。
- PyLO VeLO：[RNN 权重](https://huggingface.co/Pauljanson002/VeLO_RNN/resolve/main/model.safetensors)，9,119,908 字节；[MLP 权重](https://huggingface.co/Pauljanson002/VeLO_MLP/resolve/main/model.safetensors)，164,296 字节。各自另需 `config.json`。
- Celo：[theta.state](https://huggingface.co/amoudgl/celo/resolve/main/theta.state)，149,216 字节。
- Celo 作者转换的 Google VeLO：[theta.state](https://huggingface.co/amoudgl/velo-4000/resolve/main/theta.state)，9,282,507 字节；同目录 `params` 为 9,282,564 字节，不能混作相同文件。
- Google 默认 VeLO 路径：`gs://gresearch/learned_optimization/pretrained_lopts/aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620/params`。本轮只从官方注册表核验路径，未核验该对象当前可下载性和字节大小。

大小和哈希的调查来源为作者 Hugging Face 目录 API：[μLO](https://huggingface.co/api/models/btherien/mulo/tree/main)、[VeLO RNN](https://huggingface.co/api/models/Pauljanson002/VeLO_RNN/tree/main)、[VeLO MLP](https://huggingface.co/api/models/Pauljanson002/VeLO_MLP/tree/main)、[Celo](https://huggingface.co/api/models/amoudgl/celo/tree/main)、[转换版 VeLO](https://huggingface.co/api/models/amoudgl/velo-4000/tree/main)。仅 μLO 已在后续下载中验证二进制内容，其余仍为目录核验。

## 最小接入与参数化边界

源码入口为 [MuLO_naive.py](https://github.com/Belilovsky-Lab/pylo/blob/d05c6c1e4f4d7c46a91828a0826c2699bfe48211/pylo/optim/MuLO_naive.py)，底层为 [AdafacLO_naive.py](https://github.com/Belilovsky-Lab/pylo/blob/d05c6c1e4f4d7c46a91828a0826c2699bfe48211/pylo/optim/AdafacLO_naive.py) 与 [Meta_MLP.py](https://github.com/Belilovsky-Lab/pylo/blob/d05c6c1e4f4d7c46a91828a0826c2699bfe48211/pylo/models/Meta_MLP.py)。核心依赖为 PyTorch、NumPy、`huggingface_hub`、`safetensors`、`mup`。整包安装声明另含 `torchvision`、`torchaudio` 和 `pybind11`；不要为了这个小检查升级项目现有 PyTorch，也不要启用 `PYLO_CUDA=1` 或运行项目无关的 μP 补丁脚本。

示意调用，尚未本地执行：

```python
import torch
import mup
from pylo.optim import MuLO_naive

# model 与 base_model 必须按本次实验固定的 μP 方案创建。
mup.set_base_shapes(model, base_model)
optimizer = MuLO_naive(
    model.parameters(), lr=1.0, hf_key=local_checkpoint_directory
)
optimizer.zero_grad()
loss = observable_training_loss(model, observed_batch)
loss.backward()
optimizer.step()
```

`MuLO_naive` 比底层实现多做参数分组和宽度相关学习率缩放，并检查参数存在 `infshape`。可令基础形状与当前形状一致以进行最小加载检查，但这只让相关宽度倍率为 1，不能据此证明初始化、读出层和完整 μP 参数化已经符合原论文。若改用 `AdafacLO_naive`，无须 μP 形状标记即可消费同一预训练网络，但必须标注是改变参数化后的迁移试验，不能称原 μLO 的严格复现。μP 在方向模型上的恰当初始化需要单独核对，不能悄悄改已有基础模型后还声称同等初始能力。

## 最小验收与停止条件

1. 固定代码版本、模型卡、权重大小和内容哈希。仅加载 `safetensors` 与配置，不执行权重附带远程代码。
2. 在隔离环境先验证矩阵、向量两类参数有有限梯度和有限更新，优化器可保存恢复；记录 CPU 每步成本。不要首先接入活跃演示服务。
3. 用确定性简单损失验证短程下降，仅证明接入工作，不声称行为适应。再在完全相同的方向模型副本、相同观测和相同更新预算下比较冻结、Adam、预训练更新器。
4. 必须为在线损失写出观测来源，不能把真实扰动值、未来结果或隐藏食物/巢穴坐标当监督信号。奖励数值本身不会自动成为模型参数梯度。
5. 预训练更新器默认每次 `step` 都尝试修改有梯度的参数；没有梯度时跳过是代码条件，不是学会的更新时机。要验证用户所需“该改才改”，仍需独立门控及对应对照，不把现成优化器当作已解决全部目标。

未知项：当前 CPU/PyTorch 2.8 实际兼容性、μP 方案与当前小模型的匹配、零梯度及单标量更新稳定性、冻结参数传入时的行为、真实仿真损失的可用梯度、每只蚂蚁独立优化器状态的开销，以及在未见环境中的实际收益。以上全部需要后续执行验证，不能从分类或语言模型的公开结果外推。

## 本机接入进展

源码已固定下载至 `logs/external-models/pylo/`，未修改；原始归档 SHA-256 为 `c8a62b9c995af825fbd2f4c6cda3041ad0dbff54af59afcea561cfea1e2aadce`。原始许可证随源码保留。权重与配置位于 `logs/external-models/mulo/`，权重哈希与上述作者目录一致；配置 SHA-256 为 `03ce572db3d332984fe31817c270d486a38ee26f68277b2f9911ef8284433b94`。

最小运行入口为 `bash scripts/training/pretrained-probe.sh`，先核对文件哈希，关闭运行时 Hub 联网，以本地权重做 128 步确定性简单损失下降检查，并检查参数有限、学习型优化器自身权重不变和状态恢复后下一步相同。实际结果已保存为 `logs/external-models/mulo/cpu-probe.json`。脚本拒绝覆盖该记录，重复验收应先明确新的输出身份。

最初的 `uv run --no-sync --with` 获取遇到 TLS 错误。后续一次镜像下载恢复了连通，但该临时层递归解析了不需要的新版 CUDA PyTorch；发现后已停止对应进程，未替换项目环境。现改为 `scripts/training/pretrained-setup.sh` 将显式列出的九个小依赖以 `uv pip --no-deps --target` 安装在 `logs/external-models/runtime/`，推理脚本离线运行并复用现有 CPU PyTorch。列表固定在 `pretrained-runtime.txt`；项目现有的 NumPy、PyYAML、filelock 等依赖由原 `.venv` 提供，未改项目锁文件。

本机安装使用进程级 `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`，没有修改全局索引、代理、网络、证书校验或服务配置。实际加载已进入 128 步更新后的恢复检查，但复制模型丢失 μP 的 `infshape` 标记，接入验收暂未完整通过。错误保留在 `logs/pretrained-probe-*.log`，后续修复应只补恢复流程的形状标记，不改外部权重或优化算法。

动作基础已独立完成三个初始化的训练与验收，见[动作验收](direction-motor-acceptance.md)。这解决“用于比较的基础动作是否合格”，不解决“现成更新器能否利用局部后果改善行为”。后续优先打通外部更新器实载与可用在线损失，再决定是否训练额外的更新门控；不先从零重训整个学习型优化器。

## CPU 检查结果

第一次运行在状态恢复时因深复制不保留参数的 `infshape` 标记失败；补回形状标记并设置 `rescale_params=False`，避免恢复时改变已有参数。第二次完整运行通过：128 步，初始均方损失 `0.7376567125`，最终 `0.00016060495`，更新循环约 `0.05376` 秒；所有参数有限，预训练更新器的权重未改变，恢复后同一梯度计算产生逐位一致的下一步参数。

实际 PyTorch 路径为项目 `.venv/lib/python3.12/site-packages/torch/__init__.py`，版本 `2.8.0+cpu`。这是一个 2 输入、2 输出线性回归的接入检查，包含矩阵和向量参数，不包含单标量适应、变化环境、信号层、更新门控或蚁群。此前表格中的“未实际运行”描述调查时点；当前只对本节范围解除兼容性未知项，不外推全部模型与任务。
