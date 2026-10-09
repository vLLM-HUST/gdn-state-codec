# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from gdn_state_codec._version import __version__
from gdn_state_codec.config import parse_gdn_state_codec_experiment


def _manifest() -> dict[str, object]:
    path = (
        Path(__file__).parents[1]
        / "src/gdn_state_codec/manifests/"
        "vllm-hust-extension-v0.3.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _parse(additional_config: dict[str, object]):
    return parse_gdn_state_codec_experiment(
        additional_config,
        ["Qwen3_5ForConditionalGeneration"],
        "npu",
        True,
        None,
        0,
    )


def test_manifest03_declares_matching_manager_and_vllm_entry_points() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == "0.3-experimental"
    assert manifest["extension_id"] == "org.vllm-hust.gdn-state-codec"
    assert manifest["implementation"] == [
        {
            "type": "python_module",
            "module": "gdn_state_codec.plugin",
            "object": "register",
            "status": "active",
        }
    ]
    assert manifest["activation"]["entry_points"] == [
        {"group": "vllm.general_plugins", "name": "gdn_state_codec"}
    ]
    assert manifest["activation"]["environment"][
        "VLLM_HUST_GDN_STATE_CODEC_ENABLE"
    ] == "0"


def test_manifest_pyproject_and_package_identity_are_aligned() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = _manifest()
    assert manifest["extension_version"] == __version__
    assert project["project"]["entry-points"]["vllm.general_plugins"] == {
        "gdn_state_codec": "gdn_state_codec.plugin:register"
    }
    assert "org.vllm-hust.gdn-state-codec" in project["project"]["entry-points"][
        "vllm_hust.extension_bundles"
    ]


def test_mod_metadata_keeps_the_template_lifecycle_contract() -> None:
    metadata = json.loads(
        (Path(__file__).parents[1] / "MOD_METADATA.json").read_text(encoding="utf-8")
    )
    assert metadata["canonical_repository"] == (
        "https://github.com/vLLM-HUST/gdn-state-codec"
    )
    assert metadata["lifecycle"]["default_enabled"] is False
    assert metadata["evidence"]["qualification"] == "unqualified"


def test_manager_activation_payload_is_the_runtime_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GDN_STATE_CODEC_LIBRARY_DIR", str(tmp_path))
    activation = _manifest()["activation"]
    experiment = _parse(activation["additional_config"])
    assert experiment is not None
    assert experiment.library_dir == tmp_path
    assert experiment.mode == "shadow"
    assert experiment.initializer == "quant_only"


def test_runtime_rejects_missing_manager_library_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GDN_STATE_CODEC_LIBRARY_DIR", raising=False)
    activation = _manifest()["activation"]
    with pytest.raises(ValueError, match="GDN_STATE_CODEC_LIBRARY_DIR"):
        _parse(activation["additional_config"])
