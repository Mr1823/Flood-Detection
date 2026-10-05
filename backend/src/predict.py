"""
predict.py - classify images from the command line.

    python src/predict.py path/to/image.jpg [--threshold X]
    python src/predict.py path/to/folder     [--threshold X]   -> reports/batch_predictions.csv

For each image: the class, the flood probability, the threshold used (default:
the tuned value in models/threshold.json) and the Grad-CAM attention area.
Images go through exactly the same centred crop as evaluation (dataset.py).
"""
import argparse
import csv
import os

import config  # prints the TensorFlow version and devices; imported before TensorFlow
from dataset import load_for_model
from gradcam import attention_area, explain, load_threshold, resize_heatmap
from model import load_trained_model


def image_paths(target):
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        paths = []
        for root, dirs, files in os.walk(target):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            paths += [os.path.join(root, f) for f in sorted(files)
                      if not f.startswith(".") and f.lower().endswith(config.IMAGE_EXTENSIONS)]
        if not paths:
            raise SystemExit(f"[predict] ERROR: no images ({', '.join(config.IMAGE_EXTENSIONS)}) in {target}")
        return paths
    raise SystemExit(f"[predict] ERROR: not found: {target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Flood / no-flood prediction for an image or a folder.")
    parser.add_argument("target", help="an image file or a folder of images")
    parser.add_argument("--threshold", type=float, default=None,
                        help="decision threshold (default: the tuned value in models/threshold.json)")
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--report-dir", default=config.REPORT_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    config.set_seeds(args.seed)
    paths = image_paths(args.target)
    model = load_trained_model(os.path.join(args.model_dir, config.BEST_MODEL_FILE), compile=False)
    threshold = args.threshold if args.threshold is not None else load_threshold(args.model_dir)

    rows = []
    for path in paths:
        try:
            display, model_input = load_for_model(path)
        except Exception as exc:                  # unreadable file: report it and carry on
            print(f"{path}: could not be read ({type(exc).__name__})")
            rows.append({"file": path, "class": "unreadable", "p_flood": "", "threshold": threshold,
                         "attention_pct": ""})
            continue
        prob, cam = explain(model, model_input)
        area = attention_area(resize_heatmap(cam, display.shape[:2]))
        label = config.CLASS_NAMES[int(prob >= threshold)]
        print(f"{path}: {label.upper()}  P(flood) = {prob:.4f}  threshold = {threshold:.2f}  "
              f"attention area = {area:.1f}%")
        rows.append({"file": path, "class": label, "p_flood": round(prob, 4), "threshold": threshold,
                     "attention_pct": round(area, 1)})

    if os.path.isdir(args.target):
        config.ensure_dirs(args.report_dir)
        out = os.path.join(args.report_dir, "batch_predictions.csv")
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        n_flood = sum(r["class"] == "flood" for r in rows)
        print(f"[predict] {len(rows)} images: {n_flood} flood, {len(rows) - n_flood} other. "
              f"Saved {config.rel(out)}")


if __name__ == "__main__":
    main()
