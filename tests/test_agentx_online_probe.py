from __future__ import annotations

import json

from tools import agentx_online_probe


class _Response:
    status = 200

    def __init__(self, events: list[dict[str, object]]) -> None:
        self._lines = [
            f"data: {json.dumps(event)}\n".encode() for event in events
        ] + [b"data: [DONE]\n"]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_stream_request_marks_truncated_http_200_as_error(monkeypatch):
    response = _Response(
        [{"choices": [{"delta": {"content": "partial"}}]}]
    )
    monkeypatch.setattr(agentx_online_probe.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    result = agentx_online_probe.stream_request(
        "http://example.invalid", "model", [{"role": "user", "content": "x"}], 8, 1.0
    )

    assert result["http_status"] == 200
    assert result["content_chunk_count"] == 1
    assert result["error"].startswith("incomplete stream")


def test_stream_request_accepts_terminal_usage_and_finish(monkeypatch):
    response = _Response(
        [
            {"choices": [{"delta": {"content": "complete"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ]
    )
    monkeypatch.setattr(agentx_online_probe.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    result = agentx_online_probe.stream_request(
        "http://example.invalid", "model", [{"role": "user", "content": "x"}], 8, 1.0
    )

    assert "error" not in result
    assert result["finish_reason"] == "length"
    assert result["usage"]["completion_tokens"] == 1
