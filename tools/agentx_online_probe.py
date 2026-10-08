#!/usr/bin/env python3
"""Run a small reproducible AgentX online probe against an OpenAI endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CONVERSATIONS = (
    "4b88101f5ed7507b73138a73dfbf55f972de",
    "05f72f78a037cfa5cd4e4541960a574701d9",
    "515e43cbf9a646bb3e6489bd78f1bf1b361a",
    "16b968359e79eda8a61415fa7ad536f6cda5",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def load_first_turn(cache: Path, conversation_id: str) -> list[dict[str, Any]]:
    index = json.loads((cache / "index.dat").read_text())
    entry = index["offsets"].get(conversation_id)
    if entry is None:
        raise ValueError(f"conversation {conversation_id!r} is absent from cache")
    with (cache / "dataset.dat").open("rb") as stream:
        stream.seek(entry["offset"])
        conversation = json.loads(stream.read(entry["size"]))
    return conversation["turns"][0]["raw_messages"]


def stream_request(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint,
        data=canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    chunk_times: list[float] = []
    content: list[str] = []
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    stream_error: Any = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                stream_error = event["error"]
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
                text = choice.get("delta", {}).get("content")
                if text:
                    now = time.perf_counter()
                    if first_content_at is None:
                        first_content_at = now
                    chunk_times.append(now)
                    content.append(text)
    ended = time.perf_counter()
    if first_content_at is None:
        return {
            "http_status": status,
            "total_seconds": ended - started,
            "error": stream_error or "response contained no content chunks",
        }
    intervals_ms = [
        (right - left) * 1000
        for left, right in zip(chunk_times, chunk_times[1:])
    ]
    output = "".join(content)
    result = {
        "http_status": status,
        "total_seconds": ended - started,
        "ttft_seconds": first_content_at - started,
        "itl_median_ms": statistics.median(intervals_ms) if intervals_ms else None,
        "content_chunk_count": len(chunk_times),
        "content_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "finish_reason": finish_reason,
        "usage": usage,
    }
    if stream_error is not None:
        result["error"] = stream_error
    elif finish_reason is None or usage is None:
        result["error"] = (
            "incomplete stream: missing finish_reason or terminal usage"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18082/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--conversation-id", action="append", dest="conversation_ids")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--warmup-scope", choices=("first", "all"), default="all")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    conversation_ids = tuple(args.conversation_ids or DEFAULT_CONVERSATIONS)
    messages = {
        conversation_id: load_first_turn(args.cache, conversation_id)
        for conversation_id in conversation_ids
    }
    identities = [
        {
            "kind": "identity",
            "conversation_id": conversation_id,
            "messages_sha256": hashlib.sha256(
                canonical_json(messages[conversation_id])
            ).hexdigest(),
        }
        for conversation_id in conversation_ids
    ]

    def execute(conversation_id: str, kind: str, run: int) -> dict[str, Any]:
        try:
            result = stream_request(
                args.endpoint,
                args.model,
                messages[conversation_id],
                args.max_tokens,
                args.timeout,
            )
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        return {
            "kind": kind,
            "run": run,
            "conversation_id": conversation_id,
            **result,
        }

    records: list[dict[str, Any]] = identities
    warmup_ids = conversation_ids if args.warmup_scope == "all" else conversation_ids[:1]
    for run in range(args.warmups):
        for conversation_id in warmup_ids:
            records.append(execute(conversation_id, "warmup", run))
    jobs = [
        (conversation_id, "measurement", run)
        for run in range(args.repetitions)
        for conversation_id in conversation_ids
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(execute, *job) for job in jobs]
        for future in futures:
            records.append(future.result())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))


if __name__ == "__main__":
    main()
