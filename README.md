# GDN State Codec

GDN State Codec is a bounded Ascend 910B2 state-compression experiment for the
Qwen3.5/Qwen3.8 gated-delta recurrent state. This repository separates the
mechanism, exactness checks, workload evidence, and rejected prototypes from
the StateAxis engine integration.

## Repository metadata

- Owner: `vLLM-HUST`
- Canonical repository: `vLLM-HUST/gdn-state-codec`
- Mod ID: `org.vllm-hust.gdn-state-codec`
- Directly responsible: Shuhao Zhang (Tony) (`ShuhaoZhangTony`)
- Advisor status: confirmed none (`advisor_status: none`, `advisors: []`)
- Hardware scope: Ascend 910B2
- Default state: disabled
- Qualification: unqualified experiment; version 0.2.0 has no retained
  performance workload
- Latest candidate: quant-only vector replace; mixed sub-percent total-time
  deltas, slower TTFT, changed output hashes, and no multi-request support

`MOD_METADATA.json` is the common vLLM-HUST MOD summary. `catalog.json` is the
mechanism-specific machine-readable status record. `PROVENANCE.json` binds
the canonical repository, archived source snapshots, experiment commits, and
evidence digests. This repository is the authoritative home of the mod.

## Status

`unqualified-experiment` — disabled by default.

The active identity remains performance-unqualified: output exactness and
multi-request support did not qualify. Historical eager-only observations,
component candidates, failed gates and rejected prototypes remain separate
in [the original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/README.md). They are not inherited by the
current MOD or promoted to formal graph-mode serving claims.
See [EVIDENCE.md](EVIDENCE.md) for verified access to every original record.

## Layout

- `native/ascendc/gdn_state_codec/`: current AscendC kernels, host
  runtime, and component probes from the initial engine-integration snapshot.
- `native/include`, `native/src`, `native/tests`: portable state-capsule and
  Qwen GDN mapping reference with exactness/lifecycle tests.
- `native/specs/`: frozen mechanism and admission contracts.
- [EVIDENCE.md](EVIDENCE.md): canonical benchmark originals and integrity checks.
- `evidence/all-head-negative-m2/vectorized-prototype.patch`: retained
  rejected prototype source, not active runtime code.
- `tools/agentx_online_probe.py`: overwrite-safe streaming AgentX probe that
  rejects truncated HTTP-200 SSE responses.

## Launch with vLLM-HUST Extension Manager

The codec is an independent Manifest 0.3 vLLM plugin. It must be discovered,
configured, enabled, and launched by `vllm-hust-ext`; StateAxis is neither its
launcher nor a runtime dependency. Install the codec wheel and Extension
Manager in the same Python environment as vLLM, then provide an absolute
directory containing the matching native libraries:

```bash
vllm-hust-ext extension inspect org.vllm-hust.gdn-state-codec
vllm-hust-ext extension configure org.vllm-hust.gdn-state-codec \
  --file gdn-state-codec-manager-config.json
vllm-hust-ext extension enable org.vllm-hust.gdn-state-codec
vllm-hust-ext extension check org.vllm-hust.gdn-state-codec
vllm-hust-ext run --dry-run -- vllm serve /path/to/Qwen3.5-35B-A3B \
  --enforce-eager
vllm-hust-ext run -- vllm serve /path/to/Qwen3.5-35B-A3B --enforce-eager
```

`gdn-state-codec-manager-config.json` contains only Manager-owned overrides:

```json
{
  "environment": {
    "GDN_STATE_CODEC_LIBRARY_DIR": "/absolute/path/to/release/libraries",
    "VLLM_HUST_GDN_STATE_CODEC_ENABLE": "1"
  }
}
```

The Manifest supplies the bounded `gdn_state_codec` additional configuration
and defaults to correctness-observing `shadow` mode. ECPA discovery and even
Bundle enablement remain inert until the explicit enable variable is set;
`VLLM_HUST_GDN_STATE_CODEC_KILL_SWITCH=1` always wins. The plugin rejects a
missing or relative library directory, non-NPU devices, non-Qwen3.5 generation
architectures, graph mode, and speculative decoding before native dispatch.

## Validate the portable reference

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The AscendC component is configured separately and requires CANN 9.1.0 on
Ascend 910B2:

```bash
cmake -S native/ascendc/gdn_state_codec -B build-ascend \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DSOC_VERSION=ascend910b2
cmake --build build-ascend --parallel
```

## Provenance and integration

This repository is the canonical mod repository. The initial
engine-integration snapshot is bound to commit
`83707092ecec1eba50d4ea33ea2da615ceaa4fd8`; later experimental source and raw
evidence are bound by commit and SHA-256 in `PROVENANCE.json`. Promoting or
enabling this mod requires a new matched exactness and real-online result; the
repository name alone conveys no broader performance qualification.

Version 0.2.0 replaces the former paper-derived project identity with the
organization-owned `org.vllm-hust.gdn-state-codec` identity and
`gdn_state_codec` API surface. This is an intentional source/ABI migration, not
an alias: active integrations must update their ID, imports, library names, and
configuration. Historical evidence and archived patches remain byte-for-byte
unchanged. See `MIGRATION.md` for the exact mapping.

## Authorship and responsibility

The preserved implementation, experimental iterations, evidence curation,
and this independent mod repository are authored and directly maintained by
Shuhao Zhang (Tony) (`ShuhaoZhangTony`). No advisor role applies. This is
recorded explicitly as `advisors: []`, matching the ownership convention used
by the other mods rather than leaving advisor metadata unknown or inferred.
