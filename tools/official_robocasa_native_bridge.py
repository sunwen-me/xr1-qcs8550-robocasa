#!/usr/bin/env python3
"""Serve the official RoboCasa evaluator through an isolated XR-1 QNN context.

The evaluator sends processor tensors using the upstream pickle socket format.
This bridge rebuilds the official fixed-shape request, forwards it to the
board's native array protocol, and returns normalized actions so the official
client's processor performs its usual action decoding.
"""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_official_robocasa_requests import build_internal, numpy as to_numpy
from xr1_qwen_geometry import XR1Geometry

MAX_FRAME_BYTES = 512 * 1024 * 1024
GRAPH_ACTION_SHAPE = (1, 10, 60)


def recv_all(conn: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = conn.recv(size - len(chunks))
        if not part:
            raise ConnectionError("socket closed while receiving a frame")
        chunks.extend(part)
    return bytes(chunks)


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
        raise ValueError(f"native request is too large: {len(payload)} bytes")
    return payload


class NativeClient:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def infer(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
        payload = encode_arrays(arrays)
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as conn:
            conn.settimeout(self.timeout)
            conn.sendall(struct.pack(">I", len(payload)) + payload)
            response_size = struct.unpack(">I", recv_all(conn, 4))[0]
            if response_size > MAX_FRAME_BYTES:
                raise ValueError(f"invalid native response size: {response_size}")
            response = recv_all(conn, response_size)

        if len(response) < 12:
            raise ConnectionError("native response is truncated")
        status, action_size = struct.unpack(">II", response[:8])
        action_end = 8 + action_size
        if action_end + 4 > len(response):
            raise ValueError("native action section is malformed")
        metrics_size = struct.unpack(">I", response[action_end : action_end + 4])[0]
        metrics_start = action_end + 4
        metrics_end = metrics_start + metrics_size
        if metrics_end != len(response):
            raise ValueError("native metrics section is malformed")
        metrics = json.loads(response[metrics_start:metrics_end].decode("utf-8"))
        if status != 0:
            raise RuntimeError(f"native inference failed: {metrics}")
        expected_bytes = int(np.prod(GRAPH_ACTION_SHAPE)) * np.dtype(np.float32).itemsize
        if action_size != expected_bytes:
            raise ValueError(f"native action has {action_size} bytes; expected {expected_bytes}")
        action = np.frombuffer(response[8:action_end], dtype=np.float32).copy()
        action = action.reshape(GRAPH_ACTION_SHAPE)
        if not np.isfinite(action).all():
            raise FloatingPointError("native action contains NaN or Inf")
        return action, metrics


class OfficialBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.processor = AutoProcessor.from_pretrained(
            str(args.processor), trust_remote_code=True, use_fast=False, local_files_only=True
        )
        if self.processor.list_robot_types() != ["robocasa_mg"]:
            raise ValueError(f"unexpected processor robot types: {self.processor.list_robot_types()}")
        self.pad_id = int(self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id)
        self.geometry = XR1Geometry(processor_config=args.processor)
        self.native = NativeClient(args.native_host, args.native_port, args.native_timeout)
        self.count = 0

    def infer(self, request: dict[str, Any]) -> torch.Tensor:
        if request.get("task_id") != "robocasa_mg":
            raise ValueError(f"expected task_id=robocasa_mg, got {request.get('task_id')!r}")
        official: dict[str, Any] = {}
        for name in (
            "input_ids",
            "attention_mask",
            "image_grid_thw",
            "pixel_values",
            "state",
            "action_mask",
        ):
            if name not in request:
                raise KeyError(f"official request missing {name}")
            official[name] = to_numpy(request[name])
        official["seed"] = np.asarray([int(request.get("seed", 42))], dtype=np.int64)
        arrays = build_internal(official, self.geometry, self.pad_id, action_length=10)
        action, metrics = self.native.infer(arrays)
        if action.shape != GRAPH_ACTION_SHAPE:
            raise ValueError(f"unexpected action shape: {action.shape}")
        print(
            f"official RoboCasa request {self.count}: "
            f"request={metrics.get('request_us', -1) / 1000.0:.1f} ms, "
            f"HTP={metrics.get('npu_accel_us', -1) / 1000.0:.1f} ms, "
            f"RSS={metrics.get('rss_after_kb', -1)} kB, finite=True",
            flush=True,
        )
        self.count += 1
        # The official Client calls processor.decode_action on this tensor.
        return torch.from_numpy(action.copy())

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.args.listen_host, self.args.listen_port))
            server.listen(1)
            print(
                f"official RoboCasa native bridge listening on "
                f"{self.args.listen_host}:{self.args.listen_port}",
                flush=True,
            )
            while True:
                conn, address = server.accept()
                print(f"accepted evaluator connection from {address[0]}", flush=True)
                with conn:
                    while True:
                        try:
                            size = struct.unpack(">I", recv_all(conn, 4))[0]
                        except ConnectionError:
                            break
                        if size <= 0 or size > MAX_FRAME_BYTES:
                            raise ValueError(f"invalid evaluator frame size: {size}")
                        request = pickle.loads(recv_all(conn, size))
                        try:
                            result: Any = self.infer(request)
                        except Exception as error:
                            traceback.print_exc()
                            result = {"bridge_error": f"{type(error).__name__}: {error}"}
                        encoded = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                        conn.sendall(struct.pack(">I", len(encoded)) + encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--native-host", default="127.0.0.1")
    parser.add_argument("--native-port", type=int, default=10195)
    parser.add_argument("--native-timeout", type=float, default=900.0)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=10196)
    args = parser.parse_args()
    OfficialBridge(args).serve()


if __name__ == "__main__":
    main()
