# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from gdn_state_codec import adapter, plugin


def _clear_switches(monkeypatch):
    for name in ("ENABLE", "KILL_SWITCH", "EVIDENCE"):
        monkeypatch.delenv(plugin._var(name), raising=False)


def test_discovery_is_inert_by_default(monkeypatch) -> None:
    _clear_switches(monkeypatch)
    monkeypatch.setattr(
        adapter,
        "install",
        lambda: (_ for _ in ()).throw(AssertionError("adapter imported")),
    )
    plugin.register()


def test_kill_switch_wins(monkeypatch) -> None:
    _clear_switches(monkeypatch)
    monkeypatch.setenv(plugin._var("ENABLE"), "1")
    monkeypatch.setenv(plugin._var("KILL_SWITCH"), "1")
    monkeypatch.setattr(
        adapter,
        "install",
        lambda: (_ for _ in ()).throw(AssertionError("adapter installed")),
    )
    plugin.register()


def test_explicit_enable_installs_and_emits_evidence(monkeypatch, capsys) -> None:
    _clear_switches(monkeypatch)
    monkeypatch.setenv(plugin._var("ENABLE"), "1")
    monkeypatch.setenv(plugin._var("EVIDENCE"), "1")
    monkeypatch.setattr(
        adapter,
        "install",
        lambda: {"host": "fake.host", "mode": "deferred"},
    )
    plugin.register()
    error = capsys.readouterr().err
    assert "EVIDENCE installed mechanism=gdn_state_codec" in error
    assert "pid=" in error
