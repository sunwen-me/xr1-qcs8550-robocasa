#!/usr/bin/env python3
"""Submit exported XR-1 graph directories to AI Hub for QAIRT 2.50 compile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import qai_hub as hub

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODEL_SUFFIXES = {".onnx", ".data", ".encodings", ".bin"}


def job_id(job: Any) -> str:
    for name in ("job_id", "id"):
        value = getattr(job, name, None)
        if value:
            return str(value)
    raise RuntimeError(f"AI Hub job has no id: {job!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_hashes(path: Path) -> dict[str, str]:
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix in ALLOWED_MODEL_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"no AI Hub model files found in {path}")
    return {candidate.name: sha256(candidate) for candidate in files}


def stage_model(source_dir: Path, stage_root: Path, name: str) -> Path:
    """Expose only AI Hub model files without duplicating large ONNX data."""
    target_dir = stage_root / name
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.iterdir())
    if existing:
        return target_dir
    files = [path for path in source_dir.iterdir() if path.is_file() and path.suffix in ALLOWED_MODEL_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no AI Hub model files found in {source_dir}")
    for source in files:
        target = target_dir / source.name
        try:
            os.link(source, target)
        except OSError:
            target.symlink_to(source)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument(
        "--graph-root",
        type=Path,
        help="directory containing one subdirectory per graph; defaults to the report directory",
    )
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--device", default="QCS8550 (Proxy)")
    parser.add_argument("--qairt-version", default="2.50")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--qnn-option",
        action="append",
        default=[],
        help="append one QNN option as --qnn_options KEY=VALUE; repeatable",
    )
    parser.add_argument(
        "--extra-option",
        action="append",
        default=[],
        help="append one raw AI Hub compile option; repeatable",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        help="stage model-only directories using hard links before submission",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing job IDs in --jobs and persist each submission immediately",
    )
    args = parser.parse_args()

    report = json.loads(args.export_report.read_text(encoding="utf-8"))
    graph_root = args.graph_root or args.export_report.parent
    selected = set(args.only)
    graphs = [row for row in report["graphs"] if not selected or row["name"] in selected]
    source_hashes = {
        str(row["name"]): graph_hashes(graph_root / str(row["name"]))
        for row in graphs
    }
    report_hash = sha256(args.export_report)
    if args.resume and args.jobs.exists():
        jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
        if jobs.get("export_report_sha256") != report_hash or jobs.get("graph_source_sha256") != source_hashes:
            raise ValueError(
                "compile job state does not match the current export/source graphs; "
                "use a new jobs file or start a new release root"
            )
        jobs.setdefault("schema_version", 1)
        jobs["qairt_api_version"] = args.qairt_version
        jobs["device"] = args.device
        jobs.setdefault("graphs", {})
    else:
        jobs = {
            "schema_version": 1,
            "qairt_api_version": args.qairt_version,
            "device": args.device,
            "export_report": str(args.export_report.resolve()),
            "export_report_sha256": report_hash,
            "graph_source_sha256": source_hashes,
            "graphs": {},
        }

    def persist() -> None:
        args.jobs.parent.mkdir(parents=True, exist_ok=True)
        args.jobs.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    client = None if args.dry_run else hub.Client()
    for graph in graphs:
        name = str(graph["name"])
        existing = jobs["graphs"].get(name, {})
        if args.resume and existing.get("id"):
            print(f"{name}: already submitted ({existing['id']})", flush=True)
            continue
        model_dir = graph_root / name
        if args.stage_root:
            model_dir = stage_model(model_dir, args.stage_root, name)
        options = "--target_runtime qnn_context_binary --qairt_version " + args.qairt_version
        if name.startswith("qwen_"):
            options += " --truncate_64bit_io"
        for qnn_option in args.qnn_option:
            options += " --qnn_options " + qnn_option
        for extra_option in args.extra_option:
            options += " " + extra_option
        row: dict[str, Any] = {
            "name": name,
            "model": str(model_dir),
            "device": args.device,
            "options": options,
            "inputs": graph["inputs"],
            "outputs": graph["outputs"],
        }
        if args.dry_run:
            row["status"] = "dry_run"
        else:
            compile_job = client.submit_compile_job(
                model=str(model_dir),
                device=hub.Device(args.device),
                name=f"xr1_{name}_qcs8550_qairt250_compile",
                options=options,
            )
            row.update({"id": job_id(compile_job), "url": compile_job.url})
            print(f"{name}: {compile_job.url}", flush=True)
        jobs["graphs"][name] = row
        if args.resume:
            persist()

    persist()
    print(f"wrote {args.jobs}")


if __name__ == "__main__":
    main()
