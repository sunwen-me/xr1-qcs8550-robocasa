# Migrating an XR-1 checkpoint to QCS8550

This playbook covers the graph-export and QNN deployment path used for the published RoboCasa model. A different checkpoint can require a different processor, action contract, calibration set, graph partition, and numerical rewrite. Read those from the checkpoint and its paired training artifacts before running the pipeline.

## Inputs

Prepare the trained checkpoint, matching processor/tokenizer and configuration, action statistics and mask, representative observation/state requests, Xiaomi Robotics-1 source code, and a QCS8550-compatible QAIRT runtime. Set paths explicitly; do not copy machine paths from the example below. The released RoboCasa contract uses three 256×256 camera views, `pixel_values` `[768,1536]`, 288 Qwen tokens, a 60-value state/action vector, a 10-step action chunk, and five DiT Euler updates.

```bash
export XR1_REPO_ROOT="$PWD"
export XR1_PYTHON="$(command -v python3)"
export XR1_UPSTREAM_SOURCE="/path/to/Xiaomi-Robotics-1/xr1"
export XR1_CHECKPOINT="/path/to/checkpoint"
export XR1_PROCESSOR="/path/to/matching-processor"
export XR1_RELEASE="$XR1_REPO_ROOT/artifacts/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$XR1_RELEASE"
```

Use the same processor and statistics that were saved with the checkpoint. Do not substitute RoboCasa defaults for another task.

## Inspect and export

Inspect the checkpoint first and verify its tensor layout and dimensions against the processor and inference code.

```bash
"$XR1_PYTHON" tools/inspect_checkpoint.py "$XR1_CHECKPOINT" \
  --output "$XR1_RELEASE/checkpoint_report.json"

export XR1_UPSTREAM_SOURCE
"$XR1_PYTHON" tools/export_graphs.py \
  --checkpoint "$XR1_CHECKPOINT" \
  --processor "$XR1_PROCESSOR" \
  --output-root "$XR1_RELEASE/export" \
  --sequence-length 288 --action-length 10 --layers-per-part 6 \
  --graphs vision --graphs qwen --graphs dit \
  --robot-type robocasa_mg
```

Set action length, sequence length, camera order, robot type, and instruction to match the source model. Confirm that each graph’s input/output shapes and graph count match the intended chain.

## Prepare calibration requests

For RoboCasa, capture frames with the official evaluator environment and use the checkpoint-matched processor to build requests. Include different task families and states that span the expected input distribution.

```bash
"$XR1_PYTHON" tools/capture_robocasa_calibration_frames.py \
  --eval-root /path/to/eval_robocasa \
  --output "$XR1_RELEASE/frames" \
  --seed 0

"$XR1_PYTHON" tools/build_official_robocasa_requests.py \
  --frames "$XR1_RELEASE/frames" \
  --processor "$XR1_PROCESSOR" \
  --output "$XR1_RELEASE/calibration"
```

For a non-RoboCasa task, replace the frame and request builder with an adapter for that task’s camera/state convention. Before calibration, check image order and size, `input_ids` length, state normalization, action mask, dtype, value ranges, and finite values. Keep the validation holdout separate from calibration.

Build CPU reference chain outputs from the same internal request bundles. The reference and candidate must receive the same inputs, action noise, and sampling steps.

```bash
"$XR1_PYTHON" tools/capture_calibration_chain.py \
  --checkpoint "$XR1_CHECKPOINT" \
  --bundles "$XR1_RELEASE/calibration/internal_bundles" \
  --output "$XR1_RELEASE/calibration/chain_references"
```

## Apply and validate graph rewrites

The released RoboCasa path rewrites Vision patch embedding from Conv3D to an equivalent Gemm and applies the stable-sign SiLU identity to the first three global Qwen layers. Apply a rewrite to a copy of the export, retain its source hash, and compare output tensors on identical inputs before quantizing. The stable-sign form is

```text
u = exp(-abs(x))
r = u / (1 + u)
sigmoid = where(x >= 0, 1 - r, r)
silu = x * sigmoid
```

Its denominator stays in `[1,2]`. Use it only after confirming the exported graph contains the expected Gate/SiLU pattern; it is a model-specific graph transformation, not a universal Qwen patch.

```bash
"$XR1_PYTHON" tools/prepare_compile_models.py \
  --export-report "$XR1_RELEASE/export/export_report.json" \
  --output-root "$XR1_RELEASE/compile_source"

"$XR1_PYTHON" tools/patch_qwen_stable_sign_gate.py \
  "$XR1_RELEASE/compile_source/qwen_00_06/model.onnx" \
  "$XR1_RELEASE/compile_source/qwen_00_06/model.onnx" \
  --count 3 --report "$XR1_RELEASE/validation/qwen_00_06_gate_patch.json"

"$XR1_PYTHON" tools/patch_vision_patch_embed_gemm.py \
  --source "$XR1_RELEASE/compile_source/vision/model.onnx" \
  --output "$XR1_RELEASE/compile_source/vision/model.onnx" \
  --report "$XR1_RELEASE/validation/vision_patch_gemm.json"
```

Run `validate_vision_patch_embed_gemm.py` on the original and rewritten Vision graph using fixed inputs. Record graph hashes and output errors for every rewrite.

## Quantize and compile

Use W8A16 for the Vision patch-Gemm path and Qwen. The released DiT remains FLOAT ONNX compiled to QNN, so do not send it to the quantization job. Keep Qwen weight Q/DQ and check activation-Q/DQ, mask boundaries, and integer I/O conversion in the prepared graphs.

The QNN helper expects complete calibration internal bundles and chain references. Provide the actual graph roots and a new output directory for each run.

```bash
"$XR1_PYTHON" tools/submit_real_quantization.py \
  --graphs-root "$XR1_RELEASE/compile_source" \
  --qwen-root "$XR1_RELEASE/compile_source" \
  --dit-root "$XR1_RELEASE/compile_source" \
  --calibration-root "$XR1_RELEASE/calibration" \
  --output-root "$XR1_RELEASE/quantization" \
  --checkpoint "$XR1_CHECKPOINT" \
  --qwen-granularity six \
  --graph vision \
  --graph qwen_00_06 --graph qwen_06_12 --graph qwen_12_18 \
  --graph qwen_18_24 --graph qwen_24_30 --graph qwen_30_36
```

Prepare compile-source directories with `tools/prepare_compile_models.py`, applying any graph rewrite before QNN quantization. Compile the quantized Vision/Qwen outputs and FLOAT DiT with `tools/submit_quantized_compile.py`; set `--qairt-version` to the version supported by both compiler and board. Preserve job IDs, status, and every resulting model checksum.

## Validate on device

Install QAIRT/QNN on the board under its own license. Fetch the released contexts from the companion Hugging Face repository or assemble a new bundle from the compiled output. Run the native server for offline inference first. Use the fixed processor requests from the holdout and compare the complete action chunk against the chosen BF16 reference. Report active and full action dimensions, error formula, latency, process RSS, `MemAvailable`, temperature, and repeatability.

Only begin simulator success evaluation after the numerical comparison is complete. Keep physical robot control out of the migration loop.

## Reproducibility record

For each checkpoint, save the checkpoint and processor SHA256, software and QAIRT versions, graph partition and precision, calibration sample IDs, request/graph/context hashes, rewrite validation, fixed-request action metrics, task success counts, timing, and memory measurements. Keep all paths relative to that checkpoint’s release directory.
