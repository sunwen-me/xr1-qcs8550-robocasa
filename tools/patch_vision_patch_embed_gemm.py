#!/usr/bin/env python3
"""Replace the fixed Vision patch Conv3D with an equivalent ONNX Gemm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


PATCH_CONV = "/visual/patch_embed/proj/Conv"
PATCH_RESHAPE = "/visual/patch_embed/Reshape_1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer
    raise KeyError(f"initializer not found: {name}")


def _consumer_count(model: onnx.ModelProto, tensor_name: str) -> int:
    return sum(tensor_name in node.input for node in model.graph.node)


def _remove_node_names(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [node for node in model.graph.node if node.name not in names]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def _remove_initializer_if_dead(model: onnx.ModelProto, name: str) -> None:
    if any(name in node.input for node in model.graph.node):
        return
    kept = [initializer for initializer in model.graph.initializer if initializer.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def patch_model(source: Path, destination: Path, operator: str = "gemm") -> dict[str, object]:
    if operator not in {"gemm", "matmul", "conv2d"}:
        raise ValueError(f"unsupported operator: {operator}")
    model = onnx.load(str(source), load_external_data=True)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "pixel_values":
        raise RuntimeError("Vision graph must have one input named pixel_values")
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    if input_shape != [768, 1536]:
        raise RuntimeError(f"Vision input must be fixed [768,1536], got {input_shape}")
    nodes = list(model.graph.node)
    matches = [index for index, node in enumerate(nodes) if node.name == PATCH_CONV]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {PATCH_CONV!r}, found {len(matches)}")
    conv_index = matches[0]
    conv = nodes[conv_index]
    if conv.op_type != "Conv" or len(conv.input) != 3 or len(conv.output) != 1:
        raise RuntimeError(f"unexpected patch Conv signature: {conv}")

    weight_tensor = _initializer(model, conv.input[1])
    bias_tensor = _initializer(model, conv.input[2])
    weight = numpy_helper.to_array(weight_tensor).astype(np.float32, copy=False)
    bias = numpy_helper.to_array(bias_tensor).astype(np.float32, copy=False)
    expected_shape = (1024, 3, 2, 16, 16)
    if tuple(weight.shape) != expected_shape or tuple(bias.shape) != (1024,):
        raise RuntimeError(
            f"unexpected patch weights: weight={weight.shape}, bias={bias.shape}"
        )

    # Conv3D receives [N,C,T,H,W]. The source graph's first reshape is only
    # unpacking the already patch-packed [N,1536] input, so flattening the
    # kernel in C,T,H,W order gives the exact Gemm matrix.
    gemm_weight = weight.reshape(1024, 1536).T.copy()
    gemm_weight_name = f"visual.patch_embed.proj.{operator}_weight"
    if operator == "conv2d":
        conv2d_weight = gemm_weight.T.reshape(1024, 1536, 1, 1).copy()
        gemm_weight_tensor = numpy_helper.from_array(conv2d_weight, name=gemm_weight_name)
    else:
        gemm_weight_tensor = numpy_helper.from_array(gemm_weight, name=gemm_weight_name)
    model.graph.initializer.append(gemm_weight_tensor)

    if operator == "gemm":
        replacement_nodes = [
            helper.make_node(
                "Gemm",
                inputs=["pixel_values", gemm_weight_name, conv.input[2]],
                outputs=list(conv.output),
                name="/visual/patch_embed/proj/Gemm",
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0,
            )
        ]
    elif operator == "matmul":
        # Keep the bias as a separate broadcast Add. This is the closest
        # 2-D analogue of the original Conv3D + bias path and lets QAIRT
        # choose a different MatMul implementation than the Gemm lowering.
        matmul_output = "/visual/patch_embed/proj/MatMul_output_0"
        replacement_nodes = [
            helper.make_node(
                "MatMul",
                inputs=["pixel_values", gemm_weight_name],
                outputs=[matmul_output],
                name="/visual/patch_embed/proj/MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=[matmul_output, conv.input[2]],
                outputs=list(conv.output),
                name="/visual/patch_embed/proj/Add",
            ),
        ]
    else:
        conv2d_shape_name = "visual.patch_embed.proj.conv2d_shape"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray([768, 1536, 1, 1], dtype=np.int64),
                name=conv2d_shape_name,
            )
        )
        conv2d_input = "/visual/patch_embed/proj/Conv2D_input"
        replacement_nodes = [
            helper.make_node(
                "Reshape",
                inputs=["pixel_values", conv2d_shape_name],
                outputs=[conv2d_input],
                name="/visual/patch_embed/proj/Reshape2D",
            ),
            helper.make_node(
                "Conv",
                inputs=[conv2d_input, gemm_weight_name, conv.input[2]],
                outputs=list(conv.output),
                name="/visual/patch_embed/proj/Conv2D",
                dilations=[1, 1],
                group=1,
                kernel_shape=[1, 1],
                pads=[0, 0, 0, 0],
                strides=[1, 1],
            ),
        ]
    replacement = nodes[:conv_index] + replacement_nodes + nodes[conv_index + 1 :]
    del model.graph.node[:]
    model.graph.node.extend(replacement)

    # The old unpack/cast path is now dead. Remove only nodes proven to have
    # no other consumers; the post-projection reshape remains unchanged.
    dead_names = (
        "/visual/patch_embed/Reshape",
        "/visual/patch_embed/Cast",
        "/visual/patch_embed/proj/Constant",
    )
    removed_dead_nodes: list[str] = []
    while True:
        removed_this_pass = []
        for node_name in dead_names:
            node = next((node for node in model.graph.node if node.name == node_name), None)
            if node is not None and all(
                _consumer_count(model, output) == 0 for output in node.output
            ):
                _remove_node_names(model, {node_name})
                removed_this_pass.append(node_name)
        if not removed_this_pass:
            break
        removed_dead_nodes.extend(removed_this_pass)

    _remove_initializer_if_dead(model, conv.input[1])
    _remove_initializer_if_dead(model, conv.input[2])

    # Existing value_info entries may still describe the old 5D Conv output.
    # The tensor is reshaped immediately afterward, so letting shape inference
    # derive it from Gemm avoids carrying a stale rank through the graph.
    for collection in (model.graph.value_info,):
        kept = [value for value in collection if value.name != conv.output[0]]
        del collection[:]
        collection.extend(kept)

    model = onnx.shape_inference.infer_shapes(model)
    expected_outputs = (
        "image_features",
        "deepstack_visual_embeds_0",
        "deepstack_visual_embeds_1",
        "deepstack_visual_embeds_2",
    )
    if tuple(output.name for output in model.graph.output) != expected_outputs:
        raise RuntimeError(
            "unexpected fixed Vision output contract: "
            f"{[output.name for output in model.graph.output]}"
        )
    # The three 256x256 views contribute 3 * (256 / 16 / 2)^2 = 192
    # merged visual tokens. Gemm's symbolic first dimension is an exporter
    # artifact; the input contract above is fixed to [768,1536], so make the
    # already-reported/runtime-required output dimensions explicit for QNN.
    for output in model.graph.output:
        shape = output.type.tensor_type.shape
        del shape.dim[:]
        shape.dim.add().dim_value = 192
        shape.dim.add().dim_value = 2560
    onnx.checker.check_model(model, full_check=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.with_name(f"{destination.name}.data").unlink(missing_ok=True)
    # prepare_compile_models.py may have left the old single-file payload next
    # to the destination; do not let AI Hub upload an unreferenced stale copy.
    destination.with_name("model.data").unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{destination.name}.data",
        size_threshold=1024,
    )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "destination": str(destination),
        "destination_sha256": sha256(destination),
        "source_node": PATCH_CONV,
        "replacement_node": replacement_nodes[0].name,
        "operator": operator,
        "source_weight_shape": list(weight.shape),
        "gemm_weight_shape": list(gemm_weight.shape),
        "bias_shape": list(bias.shape),
        "input": [768, 1536],
        "patch_output_before_reshape": [768, 1024],
        "fixed_output_shapes": {name: [192, 2560] for name in expected_outputs},
        "output_shape_basis": "fixed input [768,1536], three 256x256 cameras, Qwen spatial merge 2",
        "equivalence": "Gemm(pixel_values, transpose(Conv3D.weight), bias)",
        "removed_dead_nodes": removed_dead_nodes,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--operator", choices=["gemm", "matmul", "conv2d"], default="gemm")
    args = parser.parse_args()
    report = patch_model(args.source, args.output, operator=args.operator)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
