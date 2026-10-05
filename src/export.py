"""
export.py - export the deployed model and MEASURE what deployment would cost.

The reason for choosing MobileNetV2 over ResNet or VGG is that it is small and
fast enough for a drone or a phone. This script provides the evidence - measured
numbers, never asserted ones:

  models/flood_mobilenetv2.keras         Keras model, inference only (no optimiser state)
  models/flood_mobilenetv2.tflite        TFLite, default optimisations (8-bit weights)
  models/flood_mobilenetv2_fp16.tflite   TFLite, float16 weights

For each: file size in MB, mean single-image latency over 50 runs (after warm-up
runs, which are not timed), and agreement with the Keras model on every test image.
The TFLite files run on the CPU, as they would on a phone.
Results: printed table + reports/export_benchmark.json

    python src/export.py
"""
import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np

import config  # prints the TensorFlow version and devices; imported before TensorFlow
import tensorflow as tf
from dataset import list_files, load_for_model
from gradcam import load_threshold
from model import load_trained_model

LATENCY_RUNS = 50
WARMUP_RUNS = 5


def machine_name():
    """The CPU/chip name, read from the system rather than typed in."""
    import platform
    import subprocess

    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.processor() or "unknown CPU"
    return f"{chip}, {platform.system()} {platform.release()}"


def export_inference_graph(model, out_dir):
    """Write a SavedModel whose only endpoint calls the model with training=False.

    model.export() traces the call with the training flag left at its default, and
    the random augmentation layers then stay ACTIVE in the exported graph - every
    image would be randomly rotated and re-brightened at inference. Declaring the
    endpoint explicitly guarantees an inference-only graph (and TFLite then has no
    random ops to reject).
    """
    archive = tf.keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(
        name="serve",
        fn=lambda images: model(images, training=False),
        input_signature=[tf.TensorSpec((1, *config.IMG_SIZE, config.CHANNELS), tf.float32)])
    archive.write_out(out_dir)


def convert(saved_model_dir, out_path, float16=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # weights stored as 8-bit integers ...
    if float16:
        converter.target_spec.supported_types = [tf.float16]   # ... or as 16-bit floats
    with open(out_path, "wb") as handle:
        handle.write(converter.convert())


class TFLiteModel:
    """A .tflite file with a one-image predict()."""

    def __init__(self, path):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def predict(self, image):
        self.interpreter.set_tensor(self.input["index"], image[None].astype(np.float32))
        self.interpreter.invoke()
        return float(self.interpreter.get_tensor(self.output["index"]).ravel()[0])


def time_it(fn, image):
    for _ in range(WARMUP_RUNS):                   # first calls build graphs / caches: not timed
        fn(image)
    times = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        fn(image)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.std(times))


def parse_args():
    parser = argparse.ArgumentParser(description="Export to .keras and TFLite; measure size and latency.")
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--report-dir", default=config.REPORT_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    config.set_seeds(args.seed)
    config.ensure_dirs(args.model_dir, args.report_dir)
    model = load_trained_model(os.path.join(args.model_dir, config.BEST_MODEL_FILE), compile=False)
    threshold = load_threshold(args.model_dir)     # the deployed decision threshold

    paths = {"keras": os.path.join(args.model_dir, "flood_mobilenetv2.keras"),
             "tflite": os.path.join(args.model_dir, "flood_mobilenetv2.tflite"),
             "tflite_fp16": os.path.join(args.model_dir, "flood_mobilenetv2_fp16.tflite")}
    model.save(paths["keras"])                     # compile=False above: no optimiser state saved
    saved_model_dir = tempfile.mkdtemp(prefix="flood_savedmodel_")
    try:
        export_inference_graph(model, saved_model_dir)   # augmentation switched off
        convert(saved_model_dir, paths["tflite"])
        convert(saved_model_dir, paths["tflite_fp16"], float16=True)
    finally:
        shutil.rmtree(saved_model_dir, ignore_errors=True)

    # The WHOLE test set, for timing and for checking every format gives the same answers.
    no_flood = list_files("test", "no_flood", args.data_dir)
    flood = list_files("test", "flood", args.data_dir)
    images = np.stack([load_for_model(p)[1] for p in no_flood + flood])
    labels = np.array([0] * len(no_flood) + [1] * len(flood))
    with tf.device("/CPU:0"):                      # reference probabilities: Keras on the CPU
        reference = model(images, training=False).numpy().ravel()

    keras_fn = tf.function(lambda x: model(x, training=False))
    runners = {
        "keras": ("Keras (.keras)", "Metal GPU", lambda img: keras_fn(img[None]).numpy()),
        "tflite": ("TFLite, default optimisations", "CPU", TFLiteModel(paths["tflite"]).predict),
        "tflite_fp16": ("TFLite, float16", "CPU", TFLiteModel(paths["tflite_fp16"]).predict),
    }
    results = {}
    for key, (label, device, fn) in runners.items():
        probs = np.array([float(np.ravel(fn(img))[0]) for img in images])
        mean_ms, std_ms = time_it(fn, images[0])
        agree = float(np.mean((probs >= 0.5) == (reference >= 0.5)) * 100)
        agree_tuned = float(np.mean((probs >= threshold) == (reference >= threshold)) * 100)
        pred = probs >= threshold                    # this format's own test results at the tuned threshold
        tp, fp = int((pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        results[key] = {"file": config.rel(paths[key]), "label": label, "device": device,
                        "size_mb": os.path.getsize(paths[key]) / 2**20,
                        "latency_ms_mean": mean_ms, "latency_ms_std": std_ms,
                        "max_abs_prob_diff_vs_keras_cpu": float(np.abs(probs - reference).max()),
                        "decision_agreement_pct_at_0.5": agree,
                        "decision_agreement_pct_at_tuned_threshold": agree_tuned,
                        "decisions_changed_at_tuned_threshold": int(np.sum((probs >= threshold) != (reference >= threshold))),
                        "test_at_tuned_threshold": {"tp": tp, "fp": fp, "fn": fn,
                                                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                                                    "recall": tp / (tp + fn) if tp + fn else 0.0}}

    print(f"\n{'File':<42}{'Size (MB)':>10}{'Latency (ms)':>18}  {'Device':<10}"
          f"{'max |dP| vs Keras':>18}{'same decision @0.5':>20}{f'@{threshold:.2f}':>9}")
    for r in results.values():
        print(f"{r['file']:<42}{r['size_mb']:>10.2f}{r['latency_ms_mean']:>11.2f} ± {r['latency_ms_std']:<4.2f}"
              f"  {r['device']:<10}{r['max_abs_prob_diff_vs_keras_cpu']:>18.4f}"
              f"{r['decision_agreement_pct_at_0.5']:>19.1f}%{r['decision_agreement_pct_at_tuned_threshold']:>8.1f}%")
    for r in results.values():
        t = r["test_at_tuned_threshold"]
        print(f"  {r['label']:<32} test at {threshold:.2f}: precision {t['precision']:.3f}, recall {t['recall']:.3f} "
              f"(FP {t['fp']}, FN {t['fn']}); decisions changed vs Keras: {r['decisions_changed_at_tuned_threshold']}")
    print(f"\nLatency: mean ± std of {LATENCY_RUNS} single-image runs after {WARMUP_RUNS} warm-up runs. "
          f"Agreement measured on all {len(images)} test images against Keras on the CPU.")
    benchmark = {"tuned_threshold": threshold, "latency_runs": LATENCY_RUNS, "warmup_runs": WARMUP_RUNS,
                 "agreement_images": int(len(images)), "machine": machine_name(),
                 "models": results}
    out = os.path.join(args.report_dir, "export_benchmark.json")
    with open(out, "w") as handle:
        json.dump(benchmark, handle, indent=2)
    print(f"[export] Saved {config.rel(out)}")


if __name__ == "__main__":
    main()
