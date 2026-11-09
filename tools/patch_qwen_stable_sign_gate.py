#!/usr/bin/env python3
"""Rewrite the leading Qwen Gate/SiLU sigmoid nodes to a stable-sign form."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_model(source: Path, destination: Path, count: int = 3) -> dict[str, object]:
    source_hash = sha256(source)
    model = onnx.load(str(source), load_external_data=False)
    external_files = {
        entry.value
        for tensor in model.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    for filename in external_files:
        source_data = source.parent / filename
        target_data = destination.parent / filename
        if not source_data.is_file():
            raise FileNotFoundError(source_data)
        if source_data.resolve() != target_data.resolve() and not target_data.exists():
            shutil.copy2(source_data, target_data)
    sigmoids = [node for node in model.graph.node if node.op_type == "Sigmoid"]
    if count < 1 or len(sigmoids) < count:
        raise ValueError(f"requested {count} SiLU nodes, found {len(sigmoids)}")

    for layer, sigmoid in enumerate(sigmoids[:count]):
        products = [
            node for node in model.graph.node
            if node.op_type == "Mul"
            and sigmoid.output[0] in node.input
            and sigmoid.input[0] in node.input
        ]
        if len(products) != 1:
            raise ValueError(f"Sigmoid {layer}: expected one SiLU product, found {len(products)}")

        prefix = f"xr1_stable_sign_gate_{layer}"
        names = {suffix: prefix + suffix for suffix in (
            "_one", "_zero", "_abs", "_neg_abs", "_exp", "_denom",
            "_ratio", "_one_minus", "_condition", "_sigmoid",
        )}
        model.graph.initializer.extend([
            helper.make_tensor(names["_one"], TensorProto.FLOAT, [], [1.0]),
            helper.make_tensor(names["_zero"], TensorProto.FLOAT, [], [0.0]),
        ])
        nodes = [
            helper.make_node("Abs", [sigmoid.input[0]], [names["_abs"]], name=prefix + "_Abs"),
            helper.make_node("Neg", [names["_abs"]], [names["_neg_abs"]], name=prefix + "_Neg"),
            helper.make_node("Exp", [names["_neg_abs"]], [names["_exp"]], name=prefix + "_Exp"),
            helper.make_node("Add", [names["_exp"], names["_one"]], [names["_denom"]], name=prefix + "_Add"),
            helper.make_node("Div", [names["_exp"], names["_denom"]], [names["_ratio"]], name=prefix + "_Div"),
            helper.make_node("Sub", [names["_one"], names["_ratio"]], [names["_one_minus"]], name=prefix + "_Sub"),
            helper.make_node("GreaterOrEqual", [sigmoid.input[0], names["_zero"]], [names["_condition"]], name=prefix + "_Sign"),
            helper.make_node("Where", [names["_condition"], names["_one_minus"], names["_ratio"]], [names["_sigmoid"]], name=prefix + "_Where"),
        ]
        index = next(i for i, node in enumerate(model.graph.node) if node is sigmoid)
        for offset, node in enumerate(nodes):
            model.graph.node.insert(index + offset, node)
        old_output = sigmoid.output[0]
        products[0].input[:] = [names["_sigmoid"] if value == old_output else value for value in products[0].input]
        prop = model.metadata_props.add()
        prop.key = "xr1_stable_sign_gate_global_layer"
        prop.value = str(layer)

    onnx.checker.check_model(model, full_check=False)
    onnx.save_model(model, str(destination), save_as_external_data=False)
    return {
        "source": str(source),
        "source_sha256": source_hash,
        "destination": str(destination),
        "destination_sha256": sha256(destination),
        "patched_global_layers": list(range(count)),
        "recipe": "u=exp(-abs(x)); r=u/(1+u); sigmoid=where(x>=0,1-r,r); silu=x*sigmoid",
        "external_data_files": sorted(external_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=3, help="number of leading Sigmoid/SiLU nodes to rewrite")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = patch_model(args.source, args.destination, args.count)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
