#!/usr/bin/env python3
"""Render reproducible RoboCasa camera frames for XR-1 calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = None
DEFAULT_OUTPUT = ROOT / "artifacts" / "calibration_frames"
DEFAULT_TASKS = (
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnStove",
    "TurnOnSinkFaucet",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "CoffeeServeMug",
    "TurnOnMicrowave",
    "PnPCabToCounter",
    "PnPSinkToCounter",
    "PnPMicrowaveToCounter",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "TurnOffStove",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "CoffeeSetupMug",
    "CoffeePressButton",
    "TurnOffMicrowave",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    args = parser.parse_args()

    if not 0.0 < args.crop_ratio <= 1.0:
        raise ValueError("crop ratio must be in (0, 1]")
    sys.path.insert(0, str(args.eval_root))
    from main import camera_names, center_crop_pil, create_env, render_obs

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    tasks = tuple(args.task or DEFAULT_TASKS)
    seeds = tuple(args.seed or (0,))
    for task in tasks:
        for seed in seeds:
            sample_name = f"{task}_seed{seed}"
            sample_dir = args.output / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            print(f"rendering {sample_name}", flush=True)
            env = create_env(task, seed=seed, render_onscreen=False)
            try:
                env.reset()
                controller = env.robots[0].composite_controller
                base_pos = controller.part_controllers[controller.arms[0]].origin_pos
                base_mat = controller.part_controllers[controller.arms[0]].origin_ori
                base_to_world = np.eye(4)
                base_to_world[:3, :3] = base_mat
                base_to_world[:3, 3] = base_pos
                rgbs, state, _ = render_obs(env, camera_names, base_to_world)
                images = center_crop_pil(
                    [Image.fromarray(rgb) for rgb in rgbs], args.crop_ratio
                )
                instruction = str(env.get_ep_meta()["lang"])
            finally:
                env.close()

            image_rows = []
            for index, image in enumerate(images):
                image_path = sample_dir / f"camera_{index}.png"
                image.save(image_path)
                image_rows.append(
                    {
                        "path": str(image_path),
                        "sha256": sha256(image_path),
                    }
                )
            state_path = sample_dir / "state.npy"
            np.save(state_path, np.ascontiguousarray(state, dtype=np.float32))
            metadata = {
                "sample": sample_name,
                "task": task,
                "seed": seed,
                "instruction": instruction,
                "camera_names": list(camera_names),
                "crop_ratio": args.crop_ratio,
                "images": image_rows,
                "state": {
                    "path": str(state_path),
                    "shape": list(state.shape),
                    "sha256": sha256(state_path),
                },
            }
            save_json(sample_dir / "metadata.json", metadata)
            records.append(metadata)

    save_json(
        args.output / "manifest.json",
        {
            "schema_version": 1,
            "source": "RoboCasa v0.2 simulator initial observations",
            "samples": records,
        },
    )
    print(f"captured {len(records)} samples in {args.output}")


if __name__ == "__main__":
    main()
