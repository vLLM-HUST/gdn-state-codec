# GDN State Codec windowed GDN M2

## Scope correction

The standalone boundary codec in `GDN_STATE_CODEC_QWEN38_DEVICE_CODEC_M1.md` is a
layout and correctness screen only. Its encode/copy/decode timing is not a
reproduction of LeapQuant and must not be used to accept or reject the paper's
performance claims.

M2 follows the paper's decode representation for Qwen GDN:

- `p = 16` real-token records per window;
- `r = 4` FP16 compensator records;
- an INT8 residual with FP32 row smoothing and value-channel scales;
- no dense-state materialization between boundaries;
- direct readout from the fixed boundary plus buffered rank-one records;
- power-iteration refitting only at a window boundary;
- matched comparison against the complete FP32 GDN layer kernel, never against
  an isolated device-to-device copy.

For a read vector `x`, the implementation evaluates

```text
S_l^T x = dequant(R_0)^T C Gamma(1:l) x
          + U_tilde (K_tilde^T Gamma(1:l) x)
          + sum_j u_j (k_j^T Gamma(j+1:l) x)
```

The per-token correction is `u_i = v_i - S_(i-1)^T beta_i`. The output after
the update is evaluated without forming `S_i`:

```text
o_i = S_(i-1)^T (alpha_i * q_i) + u_i * (k_i^T q_i)
```

## Qwen mapping

For the provider recurrence

```text
R <- exp(g) R
delta <- beta * (v - R k)
R <- R + delta k^T
o <- scale * R q
```

and `S = R^T`, one buffered record is:

- scalar decay `alpha = exp(g)`;
- write key `beta * k`;
- read vector `exp(g) * k`;
- correction `v - S^T read`;
- output query `scale * q`.

## Admission gates

1. CPU algebra oracle: 256 tokens, all 16 boundaries, canonical output digest
   unchanged, state and output error gates unchanged.
2. Ascend single-layer component: 32 heads, `d_k = d_v = 128`, window
   positions 0 through 15, exact lifecycle and canary checks.
3. Matched performance: identical batch, inputs, graph mode, stream policy,
   warmups, and repetitions against the FP32 GDN kernel. Report batch 1, 16,
   64, 256, and the largest admitted batch.
4. Boundary cost includes power iteration, residual smoothing, quantization,
   and record reset, amortized over exactly 16 tokens.
5. Runtime integration is prohibited unless the combined non-boundary and
   amortized boundary cost passes no-harm and produces repeatable end-to-end
   gain.

## Evidence labels

Until the authors release their announced vLLM and TileLang implementation,
all M2 results are labeled `independent-paper-reimplementation`. Component
results are not online or end-to-end results.

## Historical evidence

The [original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/native/specs/GDN_STATE_CODEC_WINDOWED_GDN_M2.md) preserves measured tables, failed gates,
source identities and limitations. These component and eager-only observations
are historical, not qualification of the current MOD. No metrics are copied here.
