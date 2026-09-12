# PDBO

This repository contains the experimental code for **Smoothing Binary
Optimization: A Primal-Dual Perspective**.

PDBO reformulates unconstrained binary optimization as a continuous minimax
problem and solves it with primal-dual gradient updates. The implementation is
written in JAX and supports batched parallel optimization on GPUs.

## Supported Problems

- Max-Cut on Gset instances
- Maximum Independent Set (MIS) on random regular graphs
- Max-k-Cut
- Max-k-SAT on 3CNF, 4CNF, and 5CNF instances

Benchmark instances are provided in `instance/`. Implementations and launch
scripts for the comparison methods are also included.

## Requirements

The main implementation requires Python, JAX, Optax, NumPy, SciPy, NetworkX,
and PyTorch. Install a JAX version compatible with your CUDA environment for
GPU experiments. Some baseline methods additionally require Gurobi and their
own dependencies.

## Running PDBO

Run commands from the repository root. For example:

```bash
# Max-Cut on Gset G1
python main.py --solver pdqubo --task mc --graph Gset --Gset_id 1 \
  --batch 100 --lr_x 0.025 --lr_y 0.025 --dual_init 6

# MIS on a random 3-regular graph
python main.py --solver pdqubo --task mis --graph reg --n 10000 --d 3 \
  --batch 10 --lr_x 0.02 --lr_y 0.02 --dual_init 5

# Max-3-Cut on Gset G1
python main.py --solver pdqubo --task maxkcut --k 3 --graph Gset \
  --Gset_id 1 --batch 100 --lr_x 0.01 --lr_y 0.01 --dual_init 6

# Max-3-SAT instance 0
python main.py --solver pdqubo --task maxsat --cnf_k 3 --cnf_id 0 \
  --batch 10 --lr_x 0.01 --lr_y 0.005 --dual_init 2
```

Complete experiment commands are available in `scripts/pdqubo/`.

## Primal Optimizer

The primal update can use standard gradient descent, RMSProp, or Adam:

```bash
python main.py --solver pdqubo --task mc --graph Gset --Gset_id 1 \
  --optimizer gd
```

Set `--optimizer` to `gd`, `rmsprop`, or `adam`. The default is `rmsprop`, which
preserves the behavior of the original experimental code. `--lr_x` controls the
primal learning rate; the dual update is unchanged.

## Reproducing Experiments

The paper evaluates Max-Cut, MIS, and Max-k-SAT using a common 180-second
budget on an NVIDIA GeForce RTX 3090. The provided shell scripts contain the
problem-specific batch sizes, learning rates, and dual initializations used by
the experiments. Results are written under `result/` when `--save True` is set.

## Citation

If this code is useful in your work, please cite:

```text
Smoothing Binary Optimization: A Primal-Dual Perspective.
```
