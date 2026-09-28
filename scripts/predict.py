"""Classify images using a saved open-set prototype memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from open_vision_memory import DinoV3FeatureExtractor, OpenSetRecognizer, UNKNOWN_LABEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Image files to classify")
    parser.add_argument("--prototype", required=True, type=Path, help="Prototype memory NPZ")
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, help="Override the saved reject threshold")
    parser.add_argument(
        "--label-map",
        type=Path,
        help='Optional JSON object mapping saved labels to display names, e.g. {"0": "airplane"}',
    )
    args = parser.parse_args()

    recognizer = OpenSetRecognizer.load(args.prototype)
    if args.threshold is not None:
        recognizer.threshold_ = args.threshold
    if recognizer.threshold_ is None:
        parser.error("prototype memory has no calibrated threshold; pass --threshold")

    selected_device = None if args.device == "auto" else args.device
    extractor = DinoV3FeatureExtractor(args.model, device=selected_device, batch_size=args.batch_size)
    images = [Image.open(path).convert("RGB") for path in args.images]
    embeddings = extractor.encode_images(images)
    predictions, distances = recognizer.predict(embeddings)

    label_map = {}
    if args.label_map:
        with args.label_map.open(encoding="utf-8") as handle:
            label_map = {str(key): str(value) for key, value in json.load(handle).items()}

    rows = []
    for path, label, distance in zip(args.images, predictions, distances, strict=True):
        rows.append({
            "image": str(path),
            "prediction": "unknown" if label == UNKNOWN_LABEL else label_map.get(str(label), str(label)),
            "prediction_id": None if label == UNKNOWN_LABEL else str(label),
            "nearest_prototype_distance": float(distance),
            "rejection_threshold": float(recognizer.threshold_),
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
