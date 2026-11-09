# Xiaomi XR-1 RoboCasa on QCS8550

This repository documents and packages an XR-1 RoboCasa policy migrated to Qualcomm QCS8550. It contains the evaluation report, result data, figures, graph-export and calibration tools, and board launch source. The compiled QNN contexts and native server binary are hosted in the companion Hugging Face model repository.

## Evaluation results

| Run | Successes | Rate |
|---|---:|---:|
| Official paper headline | — | 74.50% |
| Official episode-level records | 1,781 / 2,400 | 74.21% |
| RTX 4090 BF16 reproduction | 1,782 / 2,400 | 74.25% |
| QCS8550 QNN | 1,793 / 2,400 | 74.71% |

The runs cover 24 RoboCasa v0.2 tasks and 100 episodes per task (seeds 0–99). See [the full report](report.md) for task-level rates, figures, timing, model size, and board free-memory measurements.

## Repository contents

- `XR1_QCS8550_AGENT_PLAYBOOK.md`: checkpoint-to-board migration workflow.
- `tools/`: request construction, export, calibration, quantization, compilation, and evaluation.
- `xr1_port/`: ONNX graph wrappers and checkpoint loading helpers.
- `workbench/qnn_native_server.cpp`, `deployment/build_native.sh`: native inference-server source and cross-build recipe.
- `data/`, `figures/`, `report.md`: evaluation data and analysis.
- `LICENSE`, `NOTICE.md`: license and upstream attribution.

## Model package

Download the QNN contexts and runner from [Hugging Face](https://huggingface.co/sunwenme/xr1-robocasa-qcs8550-qnn). The package targets QCS8550 HTP v73 and QAIRT 2.50. Read its README and manifest before running it.

## Requirements and scope

Graph export requires the matching Xiaomi Robotics-1 source checkout, checkpoint, processor/tokenizer, and Python/PyTorch/Transformers/ONNX stack. Quantization and context compilation use Qualcomm AI Hub; board execution requires the matching QNN/QAIRT runtime installed separately. The released model files are compiled contexts, not the upstream training checkpoint.
