# StateAxis LeapQuant

LeapQuant is a bounded Ascend 910B2 state-compression experiment for the
Qwen3.5/Qwen3.8 gated-delta recurrent state. This repository separates the
mechanism, exactness checks, workload evidence, and rejected prototypes from
the StateAxis engine integration.

## Status

`workload-qualified-experiment` — disabled by default.

The retained v68 path passed clean-build component correctness/no-harm gates
and two matched Qwen3.5-35B/AgentX single-request eager evaluations. The first
formal bracket measured a 0.92% median end-to-end improvement and 1.09% lower
median ITL; the clean-main retest measured 1.16% and 1.69%, respectively, but
its independent bootstrap intervals overlap zero. These numbers apply only to
the recorded TP2, 649-prompt/128-output first-turn workload. They are not a
general online, graph-mode, multi-request, memory-capacity, or model-family
claim.

An expanded four-shape AgentX round did not qualify a more aggressive
boundary-refit path. A cheap preserve-compensator variant passed its component
gate and reached step 65 instead of step 22, but then exceeded the unchanged
2% real-shadow RMS limit. Exact SVD passed real-shadow correctness but was far
too expensive. No replace-mode or end-to-end speedup claim was made. See
`evidence/qwen35-agentx-expanded-20261008/` for the negative result and bound
artifact hashes.

The old 576-head boundary-scan/refit prototype failed the no-harm gate by more
than two orders of magnitude. Its patch and negative measurements remain under
`evidence/all-head-negative-m2/`; the exact archived StateAxis commit is also
preserved as an mbox patch under `archive/`. It is evidence, not active code.

## Layout

- `native/ascendc/leapquant_state_codec/`: current AscendC kernels, host
  runtime, and component probes from StateAxis main at PR #461.
- `native/include`, `native/src`, `native/tests`: portable state-capsule and
  Qwen GDN mapping reference with exactness/lifecycle tests.
- `native/specs/`: frozen mechanism and admission contracts.
- `evidence/qwen35-agentx-v68/`: summaries for the matched real-online result
  and clean-main retest.
- `evidence/qwen35-agentx-expanded-20261008/`: four-shape/concurrency probe,
  boundary-refit negative result, and immutable artifact bindings.
- `evidence/all-head-negative-m2/`: preserved rejected all-head experiment.
- `tools/agentx_online_probe.py`: overwrite-safe streaming AgentX probe that
  rejects truncated HTTP-200 SSE responses.

## Validate the portable reference

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The AscendC component is configured separately and requires CANN 9.1.0 on
Ascend 910B2:

```bash
cmake -S native/ascendc/leapquant_state_codec -B build-ascend \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DSOC_VERSION=ascend910b2
cmake --build build-ascend --parallel
```

## Provenance and integration

The canonical engine integration remains in `Qixin-Gaoke/stateaxis` merge
commit `83707092ecec1eba50d4ea33ea2da615ceaa4fd8` (PR #461). See
`PROVENANCE.json` for immutable source and evidence bindings. Promoting or
enabling this mod requires a new matched exactness and real-online result; the
repository name alone conveys no broader performance qualification.
