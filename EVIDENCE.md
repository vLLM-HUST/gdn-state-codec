# Canonical evidence

Original measurements, pairing, failures and scope limitations live in
[the frozen benchmark snapshot](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/).
This migration adds no new experiment or performance qualification.

- [Original report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/README.md)
- [Original metadata and qualification](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/MOD_METADATA.json)
- [Complete byte inventory](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/manifest.json)

EVIDENCE_SOURCES.json pins an immutable benchmark commit and the manifest hash.
Fetch originals into a cache outside this repository; every length and SHA256
is checked, and downloaded code is never executed:

```bash
python tools/benchmark_evidence.py --all --destination /tmp/gdn-state-codec-evidence
```

New measurements and Release evidence belong in benchmark. Do not commit
results or cache contents here. The source Git history remains unchanged.
