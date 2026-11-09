#!/usr/bin/env python3
"""Send fixed official RoboCasa request bundles to an XR-1 native server."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np

MAX_FRAME_BYTES = 512 * 1024 * 1024


def recv_all(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("native service closed the connection")
        data.extend(chunk)
    return bytes(data)


def encode_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    rows = []
    for name, value in arrays.items():
        encoded_name = name.encode("ascii")
        value = np.ascontiguousarray(value)
        rows.append(
            struct.pack(">H", len(encoded_name))
            + encoded_name
            + struct.pack(">Q", value.nbytes)
            + value.tobytes(order="C")
        )
    payload = struct.pack(">I", len(rows)) + b"".join(rows)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"request is too large: {len(payload)} bytes")
    return payload


def run_one(bundle: Path, host: str, port: int, timeout: float) -> tuple[np.ndarray, dict, float]:
    with np.load(bundle, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    action_shape = tuple(arrays["dit__noisy_action"].shape)
    if action_shape != (1, 10, 60):
        raise ValueError(f"unexpected official action request shape: {action_shape}")
    payload = encode_arrays(arrays)
    started = time.monotonic()
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        conn.sendall(struct.pack(">I", len(payload)) + payload)
        response_size = struct.unpack(">I", recv_all(conn, 4))[0]
        if response_size > MAX_FRAME_BYTES:
            raise ValueError(f"invalid response size: {response_size}")
        response = recv_all(conn, response_size)
    elapsed = time.monotonic() - started

    if len(response) < 12:
        raise ConnectionError("native service returned a truncated response")
    status, action_size = struct.unpack(">II", response[:8])
    action_end = 8 + action_size
    if action_end + 4 > len(response):
        raise ValueError("native response action section is malformed")
    metrics_size = struct.unpack(">I", response[action_end : action_end + 4])[0]
    metrics_start = action_end + 4
    metrics_end = metrics_start + metrics_size
    if metrics_end != len(response):
        raise ValueError("native response metrics section is malformed")
    metrics = json.loads(response[metrics_start:metrics_end].decode("utf-8"))
    expected_bytes = int(np.prod(action_shape)) * np.dtype(np.float32).itemsize
    if status != 0:
        raise RuntimeError(f"native inference returned status {status}: {metrics}")
    if action_size != expected_bytes:
        raise ValueError(f"native action is {action_size} bytes; expected {expected_bytes}")
    action = np.frombuffer(response[8:action_end], dtype=np.float32).copy().reshape(action_shape)
    return action, metrics, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10192)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    action, metrics, elapsed = run_one(args.bundle, args.host, args.port, args.timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "raw_action.npy", action)
    (args.output_dir / "native_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": 0,
        "elapsed_seconds": elapsed,
        "bundle": str(args.bundle),
        "action_shape": list(action.shape),
        "finite": bool(np.isfinite(action).all()),
        "action_min": float(np.min(action)),
        "action_max": float(np.max(action)),
        "action_l2": float(np.linalg.norm(action)),
        "metrics": metrics,
    }
    (args.output_dir / "request_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
