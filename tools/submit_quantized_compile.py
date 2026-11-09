#!/usr/bin/env python3
"""Submit the RoboCasa W8A16 quantized graphs to QNN 2.50 compile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import qai_hub as hub


ROOT = Path(__file__).resolve().parents[1]
GRAPH_NAMES = ("vision", *(f"qwen_{start:02d}_{start + 6:02d}" for start in range(0, 36, 6)), "dit_step")
TERMINAL_FAILURES = {"FAILED", "CANCELED", "ERROR"}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def job_id(job: Any) -> str:
    value = getattr(job, "job_id", None) or getattr(job, "id", None)
    if not value:
        raise RuntimeError(f"AI Hub job has no id: {job!r}")
    return str(value)


def job_status(job: Any) -> dict[str, str]:
    status = job.get_status()
    return {"state": str(status.code), "message": str(status.message or "")}


def compile_options(graph: str, qairt_version: str) -> str:
    options = f"--target_runtime qnn_context_binary --qairt_version {qairt_version}"
    if graph.startswith("qwen_"):
        options += " --truncate_64bit_io"
    return options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-root",
        type=Path,
        help="directory for downloaded context binaries; DiT defaults to <output-dir>/dit_step",
    )
    parser.add_argument("--device", default="QCS8550 (Proxy)")
    parser.add_argument("--qairt-version", default="2.50")
    parser.add_argument("--graph", action="append")
    parser.add_argument("--max-submissions", type=int, default=8)
    args = parser.parse_args()

    quantization = json.loads(args.quantization.read_text(encoding="utf-8"))
    if args.output.exists():
        state = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "qairt_api_version": args.qairt_version,
            "device": args.device,
            "quantization_jobs": str(args.quantization),
            "graphs": {},
        }
    identity = {
        "quantization_jobs": str(args.quantization.resolve()),
        "quantization_jobs_sha256": sha256(args.quantization),
        "qairt_api_version": args.qairt_version,
        "device": args.device,
    }
    if args.output.exists() and state.get("graphs") and state.get("identity") != identity:
        raise ValueError("existing quantized compile state belongs to different inputs; use a new --output")
    state["identity"] = identity
    state.setdefault("graphs", {})
    selected = set(args.graph or GRAPH_NAMES)
    unknown = selected - set(GRAPH_NAMES)
    if unknown:
        raise ValueError(f"unknown graph(s): {sorted(unknown)}")

    client = hub.Client()
    device = hub.Device(args.device)
    context_root = (args.context_root or args.output.parent / "dit_step").resolve()
    submitted = 0
    for graph in GRAPH_NAMES:
        if graph not in selected:
            continue
        source = quantization.get("graphs", {}).get(graph)
        if source is None:
            raise KeyError(f"missing quantization state for {graph}")
        qjob_id = source.get("jobs", {}).get("w8a16", {}).get("quantize_job")
        if not qjob_id:
            raise KeyError(f"missing w8a16 quantize job for {graph}")

        row = state["graphs"].setdefault(graph, {})
        row["quantize_job"] = str(qjob_id)
        quantize = hub.get_job(str(qjob_id))
        row["quantize_status"] = job_status(quantize)
        quantize_state = row["quantize_status"]["state"]
        if quantize_state in TERMINAL_FAILURES:
            row["status"] = "quantize_failed"
            print(f"{graph}: quantize {quantize_state} {row['quantize_status']['message']}", flush=True)
            continue
        if quantize_state != "SUCCESS":
            row["status"] = "waiting_quantization"
            print(f"{graph}: waiting for quantization ({quantize_state})", flush=True)
            continue

        options = compile_options(graph, args.qairt_version)
        row["options"] = options
        row["device"] = args.device
        if row.get("compile_job"):
            compile = hub.get_job(str(row["compile_job"]))
            row["compile_status"] = job_status(compile)
            row["status"] = "compile_submitted"
            if graph == "dit_step" and row["compile_status"]["state"] == "SUCCESS":
                context_root.mkdir(parents=True, exist_ok=True)
                context = context_root / "dit_step.bin"
                if not context.exists():
                    compile.download_target_model(str(context))
                row["context"] = {
                    "path": str(context),
                    "bytes": context.stat().st_size,
                    "sha256": sha256(context),
                }
            print(f"{graph}: reuse compile {row['compile_job']} ({row['compile_status']['state']})", flush=True)
            continue
        if submitted >= args.max_submissions:
            continue

        target_model = quantize.get_target_model()
        compile = client.submit_compile_job(
            model=target_model,
            device=device,
            name=f"xr1_1p5b_{graph}_w8a16_qairt250_compile",
            options=options,
        )
        row.update(
            {
                "compile_job": job_id(compile),
                "compile_url": compile.url,
                "status": "compile_submitted",
            }
        )
        submitted += 1
        save_json(args.output, state)
        print(f"{graph}: submitted {compile.job_id} {compile.url}", flush=True)

    save_json(args.output, state)
    print(f"wrote {args.output}; submitted={submitted}", flush=True)


if __name__ == "__main__":
    main()
