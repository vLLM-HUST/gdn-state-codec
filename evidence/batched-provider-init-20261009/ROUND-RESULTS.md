# Batched provider-state initialization

Date: 2026-10-09
Evidence label: `model-shaped-component-candidate`

## Bottleneck

The Qwen3.5-35B integration creates a compressed capsule lazily on the first
decode token of each of its 30 GDN layers. The retained PyTorch quant-only
initializer measured a synchronized median of `1072.90 us` per TP-local
16-state layer (`20` repetitions, minimum `1035.96 us`). This explains a
material part of the consistently worse candidate TTFT in the expanded AgentX
round.

## Candidate

Two AscendC ABI-v2 launches consume the provider-owned FP32
`[state,value,key]` tensor directly:

1. accumulate mean absolute values as 128-element UB vectors and form the
   per-key square-root smoothing vector;
2. normalize and reduce each value row as vectors, then emit the transposed
   INT8 residual layout and per-value scales expected by the decode kernel.

This removes the provider-state transpose, the PyTorch elementwise launch
chain, and intermediate normalized FP32 tensors. It does not alter the v1 ABI.

## NPU result

Both visible Ascend 910B2 devices passed the same deterministic 16-state,
128x128 probe:

| device | median two-launch time | reconstruction relative RMS |
| ---: | ---: | ---: |
| 0 | 67.990 us | 0.00738162 |
| 1 | 67.761 us | 0.00738162 |

The component launch pair is about `15.8x` lower latency than the prior
Python initializer measurement. The comparison is intentionally conservative
about scope: the Python number includes tensor allocations, while the native
probe uses preallocated outputs. StateAxis integration and matched online TTFT
must still pass before this is an end-to-end result.

The vector conversion differs from the scalar CPU rounding oracle at 4,915 of
262,144 INT8 elements because it converts through FP16, but reconstruction RMS
is below the unchanged 2% gate. The legacy v1 probe remains exact at the INT8
level (`0` mismatches) and passes with reconstruction RMS `0.0135034`.

## Decision

Admit the candidate to StateAxis shadow and matched online gates. Do not claim
an online improvement from this component result alone.
