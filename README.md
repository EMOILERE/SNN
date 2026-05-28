# SNN 项目 — DLIF / POLARA / DGN 即插即用复现

本项目实现了三种新型脉冲神经元的**即插即用**版本，都可以直接替换
`spikingjelly.activation_based.neuron.LIFNode`：

| 神经元 | 来源 | 状态 | 核心机制 |
|--------|------|------|----------|
| **DLIF** | ICLR 2026《Beyond Linear Processing》| 已开源 | 树突双线性整合 `wᵀs + sᵀKs` |
| **POLARA** | AAAI 2026《Stabilizing Spiking Neurons》| **未开源**（按论文方法复现） | 极化感知三阶段动力学 + 有界激活 |
| **DGN** | ICLR 2026《Brain-Inspired Gating》| **未开源**（按论文方法复现） | 动态电导门控（≈ LSTM forget gate）|

并基于 SpikingLeNet（3 卷积 + 2 全连接）在四种边缘感知任务上训练：
**CIFAR-10**（视觉）/ **GSC V2**（声学）/ **UCI HAR**（运动）/ **UT-HAR**（无线 CSI）。

---

## 1. 目录结构

```
snn_project/
├── neurons/
│   ├── DLIF.py              # 即插即用的 DLIF 神经元（树突双线性）
│   ├── POLARA.py            # 即插即用的 POLARA 神经元（极化感知，稳梯度）
│   ├── DGN.py               # 即插即用的 DGN 神经元（动态门控，LSTM 式）
│   └── __init__.py
├── models/
│   ├── spiking_lenet.py     # SpikingLeNet (2D) 与 SpikingLeNet1D
│   └── __init__.py
├── datasets/
│   ├── cifar10.py           # CIFAR-10（视觉）
│   ├── gsc_v2.py            # Google Speech Commands V2（声学）
│   ├── uci_har.py           # UCI HAR（运动）
│   ├── ut_har.py            # UT-HAR（无线 CSI）
│   └── __init__.py
├── scripts/
│   ├── train.py             # 训练入口
│   ├── evaluate.py          # 推理与指标分析入口
├── data/                    # 下载后的原始数据=
├── outputs/                 # 训练产物：best.pt / epoch_N.pt / history.json
├── README.md                
```

---

## 2. 环境配置

**NVIDIA GPU + CUDA 12.x** 实验环境为 RTX 4060 Laptop, Python 3.10, Pytorch 2.4.1, CUDA 12.6

### 2.1 创建 conda 环境

```bash
conda create -n snn python=3.10 -y
conda activate snn
```

### 2.2 安装依赖

```bash
# PyTorch 2.4 + CUDA 12.4
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# SNN / 数据 / 训练工具链
pip install spikingjelly==0.0.0.0.14 numpy scipy pandas scikit-learn \
    matplotlib tqdm einops h5py librosa soundfile pyyaml tensorboard timm gdown
```

## 3. 三种神经元的接口

四种神经元（含基线 LIF）都按照 SpikingJelly 多步 LIF 的接口设计，可直接互换：

```python
from neurons import DLIF, POLARA, DGN
from spikingjelly.activation_based.neuron import LIFNode

# 任选其一作为 SpikingLeNet 的神经元原型
lif    = LIFNode(tau=2.0, detach_reset=True, step_mode="m", backend="torch")
dlif   = DLIF(tau=2.0, sparsity=0.9)         # 树突双线性
polara = POLARA(window=6, gamma=1.0)         # 极化感知
dgn    = DGN(tau_s=2.0, C_init=0.5)          # 动态门控
```

**接口要点**：
- 输入张量形状 `(T, B, C, ...)`，与 `LIFNode(step_mode='m')` 完全一致；
- 都继承 `MemoryModule`，`functional.reset_net()` 自动清状态；
- DLIF / DGN 的 K / C 参数**懒初始化**，所以 `deepcopy(neuron)` 可以
  在 LeNet 的不同层得到不同通道数的实例。

---

## 4. 训练

### 4.1 基本命令模板

```bash
python scripts/train.py --task <TASK> --neuron <NEURON> [options]
```

- `<TASK>`：`cifar10 | gsc | ucihar | uthar`
- `<NEURON>`：`lif | dlif | polara | dgn`

### 4.2 完整训练命令矩阵（4 任务 × 4 神经元 = 16 种组合）

按论文与任务要求推荐的训练长度：CIFAR-10/GSC 用 30 epoch，UCI HAR / UT-HAR 用 50 epoch。

#### 4.2.1 CIFAR-10（视觉，30 epoch）

```bash
python scripts/train.py --task cifar10 --neuron lif    --epochs 30
python scripts/train.py --task cifar10 --neuron dlif   --epochs 30
python scripts/train.py --task cifar10 --neuron polara --epochs 30
python scripts/train.py --task cifar10 --neuron dgn    --epochs 30
```

#### 4.2.2 GSC V2（声学，30 epoch）

```bash
python scripts/train.py --task gsc --neuron lif    --epochs 30
python scripts/train.py --task gsc --neuron dlif   --epochs 30
python scripts/train.py --task gsc --neuron polara --epochs 30
python scripts/train.py --task gsc --neuron dgn    --epochs 30
```

#### 4.2.3 UCI HAR（运动，50 epoch）

```bash
python scripts/train.py --task ucihar --neuron lif    --epochs 50
python scripts/train.py --task ucihar --neuron dlif   --epochs 50
python scripts/train.py --task ucihar --neuron polara --epochs 50
python scripts/train.py --task ucihar --neuron dgn    --epochs 50
```

#### 4.2.4 UT-HAR（无线，50 epoch）

```bash
python scripts/train.py --task uthar --neuron lif    --epochs 50
python scripts/train.py --task uthar --neuron dlif   --epochs 50
python scripts/train.py --task uthar --neuron polara --epochs 50
python scripts/train.py --task uthar --neuron dgn    --epochs 50
```


### 4.4 所有可配置的训练参数

```bash
python scripts/train.py \
    --task     {cifar10|gsc|ucihar|uthar} \
    --neuron   {lif|dlif|polara|dgn} \
    --data-root data \
    --output-dir outputs \
    --epochs   30 \
    --batch-size 128 \
    --lr       1e-4 \
    --tau      2.0          # LIF / DLIF / DGN 都用，POLARA 不用
    --sparsity 0.9          # 仅 DLIF：K 的稀疏度
    --T        4            # 时间步数
    --hidden-dim 128        # 全连接层维度
    --num-workers 4
    --seed     0
    --quick                 # 仅跑 2 epoch（流程验证用）
    --resume   path/to.pt   # 显式指定 checkpoint（覆盖自动续训）
    --restart               # 忽略已有 last.pt，从 epoch 0 开始
    --eval-only             # 跳过训练，只跑一次测试集
    --save-every 10         # 每 N epoch 额外保存 checkpoint
```

### 4.4.1 自动断点续训（默认开启）

每 epoch 都会**原子写入**两个文件：
- `outputs/<task>_<neuron>/last.pt` — model + optimizer + epoch + history + best_acc + args
- `outputs/<task>_<neuron>/history.json` — 实时进度

**重跑同一条命令**会自动从 `last.pt` 接着上次的位置继续训练：
- Adam 优化器状态会一起恢复；
- 训练目标轮数 ≤ 已完成时直接打印 "[done]" 后退出；
- 训练目标轮数 > 已完成时从 `start_epoch = ckpt.epoch + 1` 继续。


如果 `last.pt` 里的 `task/neuron/T/hidden_dim/sparsity/tau/seed` 与当前命令
**不一致**（比如改了 `--tau`），会**警告但继续**。要换超参做消融，请加
`--restart` 或改 `--output-dir`，不要复用同一个目录。

**对 `run_all.sh` 的影响**：批量脚本里某个任务跑到一半崩了，直接重跑
`bash scripts/run_all.sh` 即可——已完成的组合会秒过，未完成的从中断处续训。


### 4.6 消融与超参实验

#### DLIF 稀疏度消融（对应论文 Table 5）

```bash
for s in 0.0 0.5 0.75 0.9 0.95; do
  python scripts/train.py --task cifar10 --neuron dlif --sparsity $s --epochs 30 \
    --output-dir outputs/dlif_sparsity_$s
done
```

#### 时间步 T 的影响

```bash
for T in 2 4 8 16; do
  python scripts/train.py --task gsc --neuron dgn --T $T --epochs 20 \
    --output-dir outputs/gsc_dgn_T$T
done
```

#### LIF 与 DLIF 的 tau 扫描

```bash
for tau in 1.5 2.0 4.0 8.0; do
  python scripts/train.py --task ucihar --neuron dlif --tau $tau --epochs 30 \
    --output-dir outputs/ucihar_dlif_tau$tau
done
```

#### Hidden dim 扫描

```bash
for h in 64 128 256 512; do
  python scripts/train.py --task ucihar --neuron polara --hidden-dim $h --epochs 30 \
    --output-dir outputs/ucihar_polara_h$h
done
```

#### 多随机种子取均值

```bash
for seed in 0 1 2; do
  python scripts/train.py --task cifar10 --neuron dlif --seed $seed --epochs 30 \
    --output-dir outputs/cifar10_dlif_seed$seed
done
```

### 4.7 继续训练 / 微调

```bash
# 同任务/同神经元继续训：直接拉长 --epochs，自动续训（无需 --resume）
python scripts/train.py --task cifar10 --neuron dlif --epochs 60

# 用某个 ckpt 做微调（fine-tune），保存到新目录避免覆盖原训练历史：
python scripts/train.py --task cifar10 --neuron dlif --epochs 20 --lr 1e-5 \
  --resume outputs/cifar10_dlif/best.pt \
  --output-dir outputs/cifar10_dlif_finetune

# 跨数据集迁移（同样的神经元在另一任务上 fine-tune）：
python scripts/train.py --task gsc --neuron dlif --epochs 20 --lr 1e-5 \
  --resume outputs/cifar10_dlif/best.pt \
  --output-dir outputs/gsc_dlif_from_cifar
```

输出位置：`outputs/<task>_<neuron>/`：
- `best.pt`：测试精度最高那次的 `{model, args, epoch, best_acc}`
- `last.pt`：每 epoch 都覆盖一次，含完整 resume 状态（model + optimizer + history + epoch + best_acc + args）
- `history.json`：每 epoch 也同步覆盖一次（实时进度跟踪）
- `epoch_N.pt`：`--save-every N` 才会产出，每 N 轮一份额外 snapshot

所有 `.pt` 文件都用**原子写入**（先写 `.tmp` 再 rename），训练中途 Ctrl-C
不会损坏 checkpoint。

---

## 5. 推理 / 评估

`evaluate.py` 在 `best.pt` 的基础上跑测试集，并输出：
- 整体 loss / 精度
- 每类精度
- 混淆矩阵
- 每层平均发放率

### 5.1 基本命令

```bash
# 用默认 ckpt: outputs/<task>_<neuron>/best.pt
python scripts/evaluate.py --task cifar10 --neuron dlif
python scripts/evaluate.py --task gsc     --neuron dgn
python scripts/evaluate.py --task ucihar  --neuron polara
python scripts/evaluate.py --task uthar   --neuron lif
```

### 5.2 指定 checkpoint

```bash
python scripts/evaluate.py --task cifar10 --neuron dlif \
  --ckpt outputs/cifar10_dlif_finetune/best.pt

# 评估特定 epoch 的中间产物（需要 train 时加 --save-every）
python scripts/evaluate.py --task gsc --neuron dgn \
  --ckpt outputs/gsc_dgn/epoch_20.pt
```

### 5.3 评估所有 16 个 checkpoint

```bash
for task in cifar10 gsc ucihar uthar; do
  for neuron in lif dlif polara dgn; do
    echo "=== $task $neuron ==="
    python scripts/evaluate.py --task $task --neuron $neuron --no-confusion
  done
done | tee outputs/all_evaluations.log
```

### 5.4 仅看测试精度（更轻量）

如果只想快速看一下当前 checkpoint 的测试精度，无需详细分析：

```bash
python scripts/train.py --task cifar10 --neuron dlif --eval-only \
  --resume outputs/cifar10_dlif/best.pt
```

---


