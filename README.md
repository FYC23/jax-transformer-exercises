# JAX Transformer Exercises

Implementation of the coding exercises in [Vlad Feinberg, *How to Land a Frontier Lab Job*](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html): a small dense transformer, a from-scratch MoE transformer, Chinchilla-style IsoFLOP sweeps for each, and a fused Pallas kernel for grouped MoE matmuls. All code is JAX + Flax dataclasses + Optax, trained on TPU.

Full writeup: [`writeup.pdf`](writeup.pdf) (LaTeX source and figures are separate). Results below are reproduced from that document.

## Task and data

Character-level 3-digit addition. Vocab size 13: digits `0`-`9`, space, `+`, `=`. Every example is a fixed length `T=16` string `a + b = c` with `a, b` in `[0, 999]` and `c = a + b`; all `10^6` pairs are enumerated and shuffled. Loss is next-token cross-entropy masked to the answer digits only. Example: `569 + 575 = 1144` with loss on `1144` alone.

## Contents

| File | Description |
| --- | --- |
| `transformer_dense.ipynb` | Dense transformer, ~8M params, pilot training run |
| `transformer_moe.ipynb` | MoE transformer, 8M active / 27M total, pilot training run |
| `transformer_dense_isoflop_sweep.ipynb` | Dense IsoFLOP sweep, 5 compute budgets |
| `transformer_moe_isoflop_sweep.ipynb` | MoE IsoFLOP sweep, 5 compute budgets, measured in active params |
| `transformer_moe_pallas_kernel.ipynb` | Fused Pallas grouped-GEMM kernel and its benchmark against `jax.lax.ragged_dot` |
| `writeup.pdf` | Full writeup with derivations, tables, and figures |

Kaggle copies: [dense](https://www.kaggle.com/code/yfrank792/jax-transformer), [MoE](https://www.kaggle.com/code/yfrank792/jax-transformer-moe), [dense sweep](https://www.kaggle.com/code/yfrank792/jax-transformer-dense-isoflop-sweep), [MoE sweep](https://www.kaggle.com/code/yfrank792/jax-transformer-moe-isoflop-sweep), [Pallas kernel](https://www.kaggle.com/code/yfrank792/jax-transformer-moe-with-fusion-kernel).

## Architecture

Decoder-only, pre-norm, residual stream: GQA attention with RoPE, GELU MLP, no gated MLP.

Parameter counts (`D` = `d_model`, `F` = `d_ff`, `L` = `n_layers`, `N`/`K` = query/KV heads, `H` = `d_qkv`, `E` = experts, `k` = active experts, `V` = vocab):

```
dense:  2D(N+K)HL + 2DFL + 2DV
total:  2D(N+K)HL + 2DEFL + 2DV + DEL
active: 2D(N+K)HL + 2DkFL + 2DV + DEL
```

Dense pilot: `L=12, D=256, F=1024, N=16, K=2, H=16, V=13`, 8.07M params.
MoE pilot: same attention stack, `E=8, k=2, F=512`, 8.09M active / 27.0M total. Routing is a linear map `W_r: D -> E`, softmax, top-k, then renormalized gates. Tokens are sorted by expert id and dispatched with `lax.ragged_dot` (custom VJP for the backward pass), so the MoE matches the dense model in active FLOPs per token.

Training: 8-chip TPU mesh, pure data parallelism over the batch axis, batch 4000 sequences (64000 tokens) chosen to sit above the ICI-bound threshold `B > CX / W_ICI ~ 35000` tokens. AdamW with cosine decay, 20 epochs, masked cross-entropy.

| Model | Eval loss @ epoch 0 | Eval loss @ epoch 19 |
| --- | --- | --- |
| Dense 8.07M (80/20 split) | 0.57 | `8.5e-4` |
| MoE 8.09M active (80/20 split) | 0.50 | `1.2e-3` |

Both models solve the task.

## IsoFLOP sweeps

Approach follows Hoffmann et al. (2022), IsoFLOP profile method: for a fixed compute budget `C`, vary model size `N`, train each on `D ~ C / 6N` tokens, and read off `N*(C)` as the argmin of eval loss. FLOPs are counted as `C = 6ND`. Runs are restricted to at most 1 epoch (`D_epoch = 0.9 * 10^6 * 16 = 1.44e7` tokens) to avoid repeating tokens. Each `(C, N)` point is an LR sweep over `{1e-4, 2e-4, ..., 8e-3}` times seeds `{42, 43, 44}`; the best mean eval loss and its seed std are recorded. Train/eval split 90/10.

Dense family (`transformer_dense_isoflop_sweep.ipynb`), `F = 4D`, 12 sizes from 5.8k to 8.1M total params. Compute budgets `C` in `{5.00, 5.95, 7.07, 8.41, 10.0}e11` (geometric grid), each slice keeping only models that fit in one epoch. Widths `D >= 80` are in the training family but underperform the smaller models and are excluded from the reported slices. Compute-optimal model sizes:

```
N*(C) = 8.8k, 12k, 8.8k, 12k, 16k params
log-log fit:  a ~ 0.71 (R^2 = 0.61),  b ~ 0.29 (R^2 = 0.13)
```

MoE family (`transformer_moe_isoflop_sweep.ipynb`): `E=8`, `k=2`, `F=2D` so active FFN FLOPs match the dense `F=4D` family. Active params and FLOPs are `N_active = N_total - N_expert(1 - k/E)` and `C = 6 N_active D`. The five budgets are one epoch of each of the five smallest widths, `C` in `{5.18, 7.80, 10.9, 14.3, 22.7}e11`; slice `j` keeps models `[j, 10]` so `n_epochs <= 1`. 11 sizes from 6.0k to 2.7M active params (13k to 8.8M total), and unlike the dense sweep the larger widths stay in the reported slices. Compute-optimal model sizes in active params:

```
N*(C) = 6.0k, 12.6k, 12.6k, 26k, 26k active params
log-log fit:  a ~ 1.0 (R^2 = 0.76),  b ~ -0.02 (R^2 ~ 0)
```

Neither sweep recovers a Chinchilla law. Slices show a shallow U at best, and the valley stays at a few thousand parameters regardless of budget instead of growing with `C`. Interpretation: the task saturates. The parametric loss form `L(N,D) = E + A/N^alpha + B/D^beta` assumes a positive irreducible entropy `E`; 3-digit addition is a small deterministic task with a simple algorithm and no entropy floor, so `A/N^alpha` collapses once the model can learn digit-wise addition and carrying. That threshold is a few thousand parameters, well below the sweep range. What the sweeps measure is how fast a given width reaches the task floor, not a compute-optimal allocation, so the fitted exponents are not meaningful and dense vs. MoE Chinchilla exponents are not identified on this task.

## Fused Pallas kernel

After routing and sorting, the MoE FFN is two ragged grouped matmuls with a GELU in between:

```
y = GELU(x W_in^(e)) W_out^(e)
```

The unfused `lax.ragged_dot` path materializes the `B_tok x F` activation in HBM between the two matmuls. The kernel fuses both grouped matmuls and the activation, so that intermediate never leaves the program: for each `F`-slice, the first dot produces a `(256, blk_F)` GELU intermediate that stays in VMEM, and the second dot contracts it into a `(256, D)` partial output.

Design: MegaBlocks-style grouped GEMM. Sorted tokens are preprocessed into tiles of size 256, with a masked tile where an expert boundary splits a full tile. The Pallas grid is `(n_tiles, F / blk_F)` with `n_tiles = B_tok / 256 + E - 1`. Each program loads a `(256, D)` activation tile and `(D, blk_F)` / `(blk_F, D)` weight slices, computes `GELU(x W_in) W_out` for that slice, then masks and read-modify-writes the partial result into the output tile. Wraps in `jax.shard_map` with a custom `lax.ragged_dot` VJP for the baseline.

Benchmark: TPU v5e-8, pure data sharding, `D = 2048`, `E = 8`, `k = 2`, sequence batch 2048, `T = 16`, `block_size = 256`, `blk_F = 512`. Activations bf16, weights fp32 cast to bf16 before the matmuls. 5 seeds per point with regenerated weights and inputs; correctness checked against `ragged_dot` (atol/rtol `1e-2`), then 100 timed trials after a `block_until_ready` compile.

| F | F/D | `ragged_dot` (s) | fused (s) | winner | ratio |
| --- | --- | --- | --- | --- | --- |
| 256 | 1/8 | `1.368e-3` | `1.515e-3` | ragged | 0.90x |
| 512 | 1/4 | `1.507e-3` | `2.025e-3` | ragged | 0.74x |
| 1024 | 1/2 | `2.093e-3` | `2.419e-3` | ragged | 0.87x |
| 2048 | 1 | `2.823e-3` | `3.318e-3` | ragged | 0.85x |
| 4096 | 2 | `4.944e-3` | `4.909e-3` | tie | 1.01x |
| 8192 | 4 | `9.791e-3` | `8.134e-3` | fused | 1.20x |
| 16384 | 8 | `2.692e-2` | `1.448e-2` | fused | 1.86x |

Seed std is `~1e-5` to `~1e-4`, so the `F=2D` point is a tie and the `F>=4D` wins are outside seed noise. `ragged_dot` wins for `F <= D`; the kernel wins clearly from `F=4D`, reaching 1.86x at `F=8D`.

Why: fusion removes the write and read of the `B_tok x F` intermediate, saving `4 B_tok F` bytes for a bf16 activation. Traffic saved as a fraction of unfused traffic is `B_tok / (2ED + B_tok + B_tok D/F)`, which grows with `F/D` and is `~19.5%` at `F=8D` (per-chip `B_tok = 2048 * 16 * 2 / 8 = 8192`). A pure bytes model at 19.5% saved predicts only `1.24x`, below the measured `1.86x`, so part of the gain comes from execution effects rather than traffic alone. Supporting evidence: doubling `F` from `4D` to `8D` doubles MLP arithmetic but raises `ragged_dot` latency by 2.75x versus 1.78x for the fused kernel. The kernel also pays preprocessing and masking overhead, which dominates at small `F` and is why it loses there.

## Running

Notebooks target TPU (`jax` + `libtpu`) and use an 8-chip mesh; the kernel benchmark additionally needs TPU-specific Pallas (`jax.experimental.pallas.tpu`). Dependencies: `jax`, `flax`, `optax`, `numpy`. The dense/MoE pilots and both sweeps are self-contained; the Pallas notebook includes the baseline MoE implementation for comparison.

```
pip install --upgrade jax libtpu flax optax
```

## Reference

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* [arXiv:2203.15556](https://arxiv.org/abs/2203.15556), 2022.
