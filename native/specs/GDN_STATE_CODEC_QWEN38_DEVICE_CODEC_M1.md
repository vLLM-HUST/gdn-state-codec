# GDN State Codec Qwen GDN device codec M1

> **Historical component only.** This standalone boundary codec does not
> implement the paper's fused `p=16` windowed decode path. Its timings must not
> be presented as a reproduction or refutation of LeapQuant. See
> `GDN_STATE_CODEC_WINDOWED_GDN_M2.md` for the corrected scope and matched gate.

## Boundary

This is a default-off Ascend 910B2 component prototype for one Qwen GDN
`128 x 128` recurrent-state head.  It converts an already fitted residual to
row-smoothed, per-value-channel symmetric INT8 and reconstructs FP32 state
with supplied rank-4 compensators.  The measured quantize path includes the
smoothing kernel.  The device screen deliberately excludes compensator
fitting, the 16-token update window, model execution, graph capture, runtime
integration, and transfer time.

The prototype must not be described as a model or serving speedup.  It is
`performance_qualified=false`, has no admission flag, and cannot replace the
unchanged FP state path.  Its purpose is to establish device exactness and a
cost floor before a batched-head kernel and matched end-to-end gate.

## Frozen mapping and layout

The provider owns state as `[value,key]`; the capsule owns its transpose as
`[key,value]`.  For provider order decay, read, update, output, the adapter is:

- `decay=exp(g)`;
- write key `beta*k`;
- read vector `exp(g)*k`;
- value `v`;
- query `scale*q`.

The CPU adapter test uses non-square `key=3,value=2` geometry and checks both
state and output equivalence so equal production dimensions cannot hide a
transpose error.

## Kernel shape

Smoothing launches one block per key row.  Quantization launches one block per
value column and uses padded UB gathers for the strided FP32 column and INT8
scatter.  Dequantization launches one block per key row.  All arithmetic is
FP32 except the stored residual codes.  The ABI is fixed to `128 x 128`, INT8,
and rank 4 for this screen.

The probe checks every smoothing value, every value scale, and all 16,384
INT8 codes against a deterministic host oracle.  It then reports relative RMS
against residual plus the supplied rank-4 product.  This RMS is quantization
error, not model-quality evidence.

## Historical evidence

The [original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/native/specs/GDN_STATE_CODEC_QWEN38_DEVICE_CODEC_M1.md) preserves measured tables, failed gates,
source identities and limitations. These component and eager-only observations
are historical, not qualification of the current MOD. No metrics are copied here.
