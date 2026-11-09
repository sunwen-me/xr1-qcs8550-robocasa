#!/usr/bin/env python3
"""Submit resumable real-chain W8A16/W4A16 jobs for the XR1 1.5B export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import qai_hub as hub
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[1]
PRECISIONS = {
    "w8a16": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT16),
    "w4a16": (hub.QuantizeDtype.INT4, hub.QuantizeDtype.INT16),
}
INTEGER_INPUTS = {"input_ids", "attention_mask", "position_ids"}
ONE_LAYER_QWEN_GRAPHS = tuple(f"qwen_{layer:02d}_{layer + 1:02d}" for layer in range(36))
SIX_LAYER_QWEN_GRAPHS = tuple(f"qwen_{start:02d}_{start + 6:02d}" for start in range(0, 36, 6))


def discover_qwen_graphs(root: Path, granularity: str) -> tuple[str, ...]:
    """Discover one-layer or six-layer Qwen source graphs without guessing."""
    candidates = []
    for path in root.glob("qwen_*_*/model.onnx"):
        parts = path.parent.name.split("_")
        if len(parts) != 3 or not all(part.isdigit() for part in parts[1:]):
            continue
        start, end = (int(part) for part in parts[1:])
        if end <= start or start < 0 or end > 36:
            continue
        candidates.append((start, end, path.parent.name))
    candidates.sort()
    names = tuple(name for _start, _end, name in candidates)
    if granularity == "one":
        expected = ONE_LAYER_QWEN_GRAPHS
    elif granularity == "six":
        expected = SIX_LAYER_QWEN_GRAPHS
    elif granularity == "auto":
        widths = {end - start for start, end, _name in candidates}
        if len(widths) != 1:
            raise ValueError(f"mixed or missing Qwen graph widths under {root}: {names}")
        width = next(iter(widths), 0)
        expected = ONE_LAYER_QWEN_GRAPHS if width == 1 else SIX_LAYER_QWEN_GRAPHS if width == 6 else ()
    else:
        raise ValueError(f"unsupported Qwen granularity: {granularity}")
    if names != expected:
        raise FileNotFoundError(
            f"expected {granularity} Qwen graphs under {root}, got {list(names)}; "
            "check --qwen-root and --qwen-granularity"
        )
    return expected


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            with child.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(name: str) -> str:
    return "hidden_states" if name.startswith("hidden_states.") else name


def dtype_for_input(value_info: onnx.ValueInfoProto) -> np.dtype:
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type in (TensorProto.INT32, TensorProto.INT64):
        return np.dtype(np.int32)
    if elem_type == TensorProto.FLOAT:
        return np.dtype(np.float32)
    raise TypeError(f"unsupported ONNX input type {elem_type} for {value_info.name}")


def input_names_and_dtypes(graph_dir: Path) -> tuple[list[str], dict[str, np.dtype]]:
    model = onnx.load(str(graph_dir / "model.onnx"), load_external_data=False)
    names = [value.name for value in model.graph.input]
    return names, {value.name: dtype_for_input(value) for value in model.graph.input}


def graph_source_hashes(graph_names: tuple[str, ...], graphs_root: Path, qwen_root: Path, dit_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for graph in ("vision", *graph_names, "dit_step"):
        root = qwen_root if graph.startswith("qwen_") else dit_root if graph == "dit_step" else graphs_root
        graph_dir = root / graph
        model_path = graph_dir / "model.onnx"
        model = onnx.load(str(model_path), load_external_data=False)
        external_locations = {
            str(item.value)
            for initializer in model.graph.initializer
            for item in initializer.external_data
            if item.key == "location"
        }
        files = [
            model_path,
            *sorted(graph_dir.glob("*.data")),
            *(graph_dir / location for location in sorted(external_locations)),
        ]
        files = list(dict.fromkeys(files))
        files = [path for path in files if path.is_file()]
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if external_locations and any(not (graph_dir / location).is_file() for location in external_locations):
            missing = sorted(location for location in external_locations if not (graph_dir / location).is_file())
            raise FileNotFoundError(f"{graph}: missing ONNX external data: {missing}")
        unreferenced = sorted(
            path.name for path in graph_dir.glob("*.data") if path.name not in external_locations
        )
        if unreferenced:
            raise ValueError(f"{graph}: unreferenced external data files would be uploaded: {unreferenced}")
        result[graph] = {path.name: sha256(path) for path in files}
    return result


def samples(calibration_root: Path) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]]:
    bundle_manifest = json.loads(
        (calibration_root / "internal_bundles/manifest.json").read_text(encoding="utf-8")
    )
    reference_manifest = json.loads(
        (calibration_root / "chain_references/manifest.json").read_text(encoding="utf-8")
    )
    references = {str(row["sample"]): Path(str(row["path"])) for row in reference_manifest["samples"]}
    result = []
    for row in bundle_manifest["samples"]:
        sample = str(row["sample"])
        bundle_path = Path(str(row["path"]))
        reference_path = references.get(sample)
        if reference_path is None:
            raise KeyError(f"missing chain reference for {sample}")
        with np.load(bundle_path, allow_pickle=False) as archive:
            bundle = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        with np.load(reference_path, allow_pickle=False) as archive:
            reference = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        result.append((bundle, reference, sample))
    if not result:
        raise ValueError("calibration set is empty")
    return result


def typed(name: str, value: np.ndarray, dtypes: dict[str, np.dtype]) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=dtypes[name])


def qwen_calibration(
    graph: str, input_names: list[str], dtypes: dict[str, np.dtype], rows: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]]
) -> dict[str, list[np.ndarray]]:
    start = int(graph.split("_")[1])
    result = {name: [] for name in input_names}
    for bundle, reference, _sample in rows:
        logical = {name[len("qwen__") :]: value for name, value in bundle.items() if name.startswith("qwen__")}
        mask = np.asarray(logical["visual_pos_mask"], dtype=bool).squeeze(-1)
        for source, target in (
            ("image_features", "vision_embeds_padded"),
            ("deepstack_visual_embeds_0", "deepstack_visual_embeds_0_padded"),
            ("deepstack_visual_embeds_1", "deepstack_visual_embeds_1_padded"),
            ("deepstack_visual_embeds_2", "deepstack_visual_embeds_2_padded"),
        ):
            padded = np.array(logical[target], dtype=np.float32, copy=True)
            padded[mask] = reference[f"vision__{source}"]
            logical[target] = padded
        if start:
            previous = f"qwen_{start - 1:02d}_{start:02d}__hidden_states"
            logical["hidden_states"] = reference[previous]
        for name in input_names:
            key = canonical(name)
            if key not in logical:
                raise KeyError(f"{graph}: missing {key} in {sorted(logical)}")
            result[name].append(typed(name, logical[key], dtypes))
    return result


def vision_calibration(
    input_names: list[str], dtypes: dict[str, np.dtype], rows: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]]
) -> dict[str, list[np.ndarray]]:
    result = {name: [] for name in input_names}
    for bundle, _reference, _sample in rows:
        logical = {name[len("vision__") :]: value for name, value in bundle.items() if name.startswith("vision__")}
        for name in input_names:
            result[name].append(typed(name, logical[name], dtypes))
    return result


def dit_calibration(
    input_names: list[str], dtypes: dict[str, np.dtype], rows: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]]
) -> dict[str, list[np.ndarray]]:
    result = {name: [] for name in input_names}
    for bundle, reference, _sample in rows:
        fixed = {
            name: np.ascontiguousarray(bundle[f"dit__{name}"], dtype=np.float32)
            for name in ("action_mask", "state", "cos", "sin", "attention_mask")
        }
        fixed["k_cache"] = np.ascontiguousarray(reference["k_cache"], dtype=np.float32)
        fixed["v_cache"] = np.ascontiguousarray(reference["v_cache"], dtype=np.float32)
        action = np.ascontiguousarray(bundle["dit__noisy_action"], dtype=np.float32)
        for step in range(5):
            values = dict(fixed)
            values["noisy_action"] = action.copy()
            values["timestep"] = np.asarray(step / 5.0, dtype=np.float32).reshape(1, 1, 1)
            for name in input_names:
                result[name].append(typed(name, values[name], dtypes))
            action = action + np.ascontiguousarray(reference[f"velocity_step_{step}"], dtype=np.float32) / 5.0
    return result


def data_summary(data: dict[str, list[np.ndarray]]) -> dict[str, object]:
    return {
        name: {
            "samples": len(values),
            "shape": list(values[0].shape),
            "dtype": str(values[0].dtype),
            "finite": bool(all(np.isfinite(value).all() for value in values)),
        }
        for name, values in data.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-root", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--dit-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint recorded in the release provenance",
    )
    parser.add_argument(
        "--qwen-granularity",
        choices=("auto", "one", "six"),
        default="auto",
        help="Qwen source graph width; auto requires a complete one- or six-layer set",
    )
    parser.add_argument("--device", default="QCS8550 (Proxy)")
    parser.add_argument("--precision", action="append", choices=tuple(PRECISIONS), default=["w8a16"])
    parser.add_argument("--graph", action="append")
    args = parser.parse_args()

    if args.checkpoint and not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    qwen_graphs = discover_qwen_graphs(args.qwen_root, args.qwen_granularity)
    source_granularity = "one" if qwen_graphs == ONE_LAYER_QWEN_GRAPHS else "six"
    graph_names = ("vision", *qwen_graphs, "dit_step")
    source_hashes = graph_source_hashes(qwen_graphs, args.graphs_root, args.qwen_root, args.dit_root)
    selected = list(args.graph or graph_names)
    unknown = set(selected) - set(graph_names)
    if unknown:
        raise ValueError(f"unknown graph(s) for {args.qwen_granularity}-layer source: {sorted(unknown)}")
    rows = samples(args.calibration_root)
    jobs_path = args.output_root / "quantization_jobs.json"
    checkpoint_sha = sha256(args.checkpoint) if args.checkpoint and args.checkpoint.exists() else ""
    calibration_manifests = {
        name: args.calibration_root / name / "manifest.json"
        for name in ("internal_bundles", "chain_references")
    }
    calibration_sha = {
        name: sha256(path) if path.is_file() else ""
        for name, path in calibration_manifests.items()
    }
    state = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "graphs_root": str(args.graphs_root.resolve()),
        "qwen_root": str(args.qwen_root.resolve()),
        "dit_root": str(args.dit_root.resolve()),
        "qwen_graphs": list(qwen_graphs),
        "qwen_granularity": source_granularity,
        "calibration_root": str(args.calibration_root.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "calibration_manifest_sha256": calibration_sha,
        "source_graph_sha256": source_hashes,
        "samples": [sample for _bundle, _reference, sample in rows],
        "graphs": {},
    }
    if jobs_path.exists() and args.checkpoint and state.get("graphs"):
        if (
            not state.get("checkpoint_sha256")
            or not state.get("calibration_manifest_sha256")
            or not state.get("source_graph_sha256")
        ):
            raise ValueError(
                "existing quantization state lacks release provenance; use a new --output-root"
            )
    for key, expected in (
        ("checkpoint_sha256", checkpoint_sha),
        ("calibration_manifest_sha256", calibration_sha),
        ("calibration_root", str(args.calibration_root.resolve())),
        ("graphs_root", str(args.graphs_root.resolve())),
        ("qwen_root", str(args.qwen_root.resolve())),
        ("dit_root", str(args.dit_root.resolve())),
        ("source_graph_sha256", source_hashes),
    ):
        actual = state.get(key)
        if actual and actual != expected:
            raise ValueError(f"quantization state provenance mismatch for {key}: {actual!r} != {expected!r}")
    existing_graphs = tuple(state.get("qwen_graphs", ()))
    if existing_graphs and existing_graphs != qwen_graphs:
        raise ValueError(
            "quantization state uses a different Qwen graph set; "
            f"state={list(existing_graphs)} current={list(qwen_graphs)}"
        )
    state.setdefault("graphs_root", str(args.graphs_root.resolve()))
    state.setdefault("qwen_root", str(args.qwen_root.resolve()))
    state.setdefault("dit_root", str(args.dit_root.resolve()))
    state.setdefault("calibration_root", str(args.calibration_root.resolve()))
    state.setdefault("qwen_graphs", list(qwen_graphs))
    state.setdefault("qwen_granularity", source_granularity)
    state.setdefault("checkpoint_sha256", checkpoint_sha)
    state.setdefault("calibration_manifest_sha256", calibration_sha)
    state.setdefault("source_graph_sha256", source_hashes)
    if args.checkpoint:
        state["checkpoint"] = str(args.checkpoint)
    device = hub.Device(args.device)

    for graph in selected:
        if graph.startswith("qwen_"):
            graph_dir = args.qwen_root / graph
        elif graph == "dit_step":
            graph_dir = args.dit_root / graph
        else:
            graph_dir = args.graphs_root / graph
        input_names, dtypes = input_names_and_dtypes(graph_dir)
        if graph == "vision":
            calibration = vision_calibration(input_names, dtypes, rows)
        elif graph == "dit_step":
            calibration = dit_calibration(input_names, dtypes, rows)
        else:
            calibration = qwen_calibration(graph, input_names, dtypes, rows)
        if not all(row["finite"] for row in data_summary(calibration).values()):
            raise ValueError(f"non-finite calibration data for {graph}")

        graph_state = state["graphs"].setdefault(
            graph,
            {
                "model_path": str(graph_dir),
                "inputs": data_summary(calibration),
                "jobs": {},
            },
        )
        model_id = graph_state.get("model_id")
        model = hub.get_model(str(model_id)) if model_id else hub.upload_model(str(graph_dir), name=f"xr1_robocasa_{graph}_fp32_source")
        if not model_id:
            graph_state["model_id"] = model.model_id
            save_json(jobs_path, state)
            print(f"{graph}: uploaded source {model.model_id}", flush=True)

        dataset_ids = graph_state.setdefault("dataset_ids", {})
        for precision in args.precision:
            job_state = graph_state["jobs"].get(precision, {})
            if job_state.get("quantize_job"):
                print(f"{graph} {precision}: reuse {job_state['quantize_job']}", flush=True)
                continue
            dataset_id = dataset_ids.get(precision)
            dataset = hub.get_dataset(str(dataset_id)) if dataset_id else hub.upload_dataset(
                calibration, name=f"xr1_robocasa_{graph}_{precision}_calibration"
            )
            if not dataset_id:
                dataset_ids[precision] = dataset.dataset_id
                save_json(jobs_path, state)
                print(f"{graph} {precision}: uploaded dataset {dataset.dataset_id}", flush=True)
            weights_dtype, activations_dtype = PRECISIONS[precision]
            job = hub.submit_quantize_job(
                model=model,
                calibration_data=dataset,
                weights_dtype=weights_dtype,
                activations_dtype=activations_dtype,
                name=f"xr1_robocasa_{graph}_{precision}_quantize",
            )
            graph_state["jobs"][precision] = {
                "quantize_job": job.job_id,
                "quantize_url": job.url,
                "status": "SUBMITTED",
            }
            save_json(jobs_path, state)
            print(f"{graph} {precision}: submitted {job.job_id} {job.url}", flush=True)

    save_json(jobs_path, state)
    print(f"wrote {jobs_path}")


if __name__ == "__main__":
    main()
