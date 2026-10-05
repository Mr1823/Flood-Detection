"""
export_dashboard_data.py - collect everything in reports/ into one JSON file for the
React results dashboard.

The dashboard is a static, read-only view: React cannot run the Keras model, so every
number it shows must already have been produced by the pipeline. This script only
reads, reshapes and copies. It never recomputes a metric and never invents one - if a
value is missing here, the script that produces it is the thing to fix.

Sources (all required; a missing one is a hard error)
  reports/metrics.json              threshold, confusion matrices, AUC, counts
  reports/seed_summary.json         frozen vs fine-tuned, mean +/- std over 3 seeds
  reports/history.json              per-epoch training history for the deployed run
  reports/curves.json               ROC and PR curve points        (from evaluate.py)
  reports/test_probabilities.csv    P(flood) per test image        (from evaluate.py)
  reports/threshold_sweep_val.csv   the sweep the threshold is chosen from
  reports/threshold_sweep_test.csv  the same sweep on test, reported only
  reports/export_benchmark.json     TFLite size / latency / agreement
  reports/gradcam_stats.json        attention area and the deletion test
  reports/audit_split.json          dataset validation - the training split
  reports/audit_rejected.json       dataset validation - the rejected dataset
  reports/audit_aider.json          dataset validation - AIDER itself

Output
  dashboard/public/data/dashboard.json

    python src/export_dashboard_data.py
"""
import argparse
import csv
import json
import os
from datetime import datetime

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "dashboard", "public", "data", "dashboard.json")


# ---------------------------------------------------------------------------
# Loading - loudly
# ---------------------------------------------------------------------------
def require(report_dir, name):
    """Load a required report file, or fail with the command that produces it."""
    path = os.path.join(report_dir, name)
    if not os.path.isfile(path):
        produced_by = {
            "metrics.json": "python src/evaluate.py",
            "curves.json": "python src/evaluate.py",
            "test_probabilities.csv": "python src/evaluate.py",
            "threshold_sweep_val.csv": "python src/evaluate.py",
            "threshold_sweep_test.csv": "python src/evaluate.py",
            "history.json": "python src/train.py",
            "seed_summary.json": "python src/train.py --seeds 42 43 44",
            "export_benchmark.json": "python src/export.py",
            "gradcam_stats.json": "python src/gradcam.py",
            "audit_split.json": "python src/audit_dataset.py data",
            "audit_rejected.json": "python src/audit_dataset.py data/rejected",
            "audit_aider.json": "python src/audit_dataset.py aider_raw/AIDER",
        }.get(name, "the pipeline")
        raise SystemExit(f"[export] MISSING: {path}\n"
                         f"[export] Produce it first with: {produced_by}")
    if path.endswith(".json"):
        with open(path) as handle:
            return json.load(handle)
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Reshaping - each function returns one top-level key
# ---------------------------------------------------------------------------
def class_order(metrics, split_audit):
    """[negative, positive] - the order the confusion matrix rows and columns use.

    Derived, not assumed: the positive class is whichever test class has the flood
    count that metrics.json reports, so this keeps working if the classes are renamed.
    """
    test_counts = split_audit["counts"]["test"]
    positive = [name for name, n in test_counts.items() if n == metrics["test_floods"]]
    if len(positive) != 1:
        raise SystemExit(f"[export] Cannot tell which test class is the positive one from "
                         f"{test_counts} and test_floods={metrics['test_floods']}.")
    negative = [name for name in test_counts if name != positive[0]]
    return [negative[0], positive[0]]


def build_meta(metrics, history, split_audit, threshold):
    """Header facts: what model, which seed, how much data."""
    counts = split_audit["counts"]                     # {"train": {"flood": n, ...}, ...}
    return {
        "class_names": class_order(metrics, split_audit),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_name": metrics["model"],
        "architecture": "MobileNetV2 (ImageNet weights), frozen then fine-tuned",
        "seed_deployed": history["deployed"]["seed"],
        "variant_deployed": history["deployed"]["variant"],
        "selection_rule": history["deployed"]["rule"],
        "threshold": threshold,
        "threshold_rationale": metrics["threshold"]["rationale"],
        "threshold_selected_on": metrics["threshold"]["selected_on"],
        "n_train": sum(counts["train"].values()),
        "n_val": sum(counts["val"].values()),
        "n_test": sum(counts["test"].values()),
        "class_counts": {split: dict(by_class) for split, by_class in counts.items()},
    }


def build_comparison(seed_summary):
    """Frozen vs fine-tuned on the test set, mean +/- std across the 3 seeds."""
    labels = {"test_accuracy": "accuracy", "test_auc": "AUC",
              "test_precision": "precision", "test_recall": "recall"}
    rows = []
    for variant, metrics in seed_summary["variants"].items():
        for key, label in labels.items():
            rows.append({"variant": variant, "metric": label,
                         "mean": metrics[key]["mean"], "std": metrics[key]["std"],
                         "values": metrics[key]["values"]})
    return rows


def build_history(history):
    """Wide per-epoch history -> long format: one row per epoch/split/metric."""
    h = history["history"]
    rows = []
    for metric in ("accuracy", "loss", "auc"):
        for i, epoch in enumerate(h["epoch"]):
            rows.append({"epoch": epoch, "phase": h["phase"][i], "split": "train",
                         "metric": metric, "value": h[metric][i]})
            rows.append({"epoch": epoch, "phase": h["phase"][i], "split": "validation",
                         "metric": metric, "value": h["val_" + metric][i]})
    return rows


def build_sweep(val_rows, test_rows):
    """Both threshold sweeps, tagged by split. Validation is where the choice was made."""
    rows = []
    for split, source in (("validation", val_rows), ("test", test_rows)):
        for row in source:
            rows.append({"split": split, "threshold": float(row["threshold"]),
                         "precision": float(row["precision"]), "recall": float(row["recall"]),
                         "f1": float(row["f1"])})
    return rows


def build_confusion(metrics):
    """[[TN, FP], [FN, TP]] at the default and the tuned threshold."""
    def matrix(block):
        return [[block["tn"], block["fp"]], [block["fn"], block["tp"]]]

    return {"default": matrix(metrics["test_at_default_0.5"]),
            "tuned": matrix(metrics["test_at_tuned_threshold"]),
            "default_threshold": metrics["test_at_default_0.5"]["threshold"],
            "tuned_threshold": metrics["test_at_tuned_threshold"]["threshold"],
            "note": metrics["confusion_matrix_note"]}


def build_probabilities(rows):
    """P(flood) for every test image, with the class it actually is."""
    return [{"true_class": row["true_class"], "p_flood": float(row["p_flood"])} for row in rows]


def build_export(benchmark):
    """One row per exported format: how big, how fast, does it still agree."""
    rows = []
    for model in benchmark["models"].values():
        rows.append({
            "format": model["label"],
            "device": model["device"],
            "size_mb": model["size_mb"],
            "latency_ms": model["latency_ms_mean"],
            "latency_std_ms": model["latency_ms_std"],
            "agrees_with_keras": model["decision_agreement_pct_at_tuned_threshold"],
            "decisions_changed": model["decisions_changed_at_tuned_threshold"],
        })
    return rows


def build_gradcam(stats):
    """Attention area on floods, and the deletion test that is the real evidence."""
    deletion = stats["deletion_test_detected_floods"]
    return {
        "layer": stats["layer"],
        "mean_attention_flood": stats["true_flood"]["mean_attention_pct"],
        "mean_attention_no_flood": stats["true_no_flood"]["mean_attention_pct"],
        "pct_above_50": stats["true_flood"]["share_above_warn_pct"],
        "warn_above_pct": stats["warn_above_pct"],
        "deletion": {
            "images": deletion["images"],
            "deleted_share": deletion["deleted_share"],
            "baseline": deletion["mean_p_flood_original"],
            "most_attended_removed": deletion["mean_p_flood_most_attended_deleted"],
            "least_attended_removed": deletion["mean_p_flood_least_attended_deleted"],
            "still_flood_most_removed_pct": deletion["still_flood_after_most_attended_deleted_pct"],
            "still_flood_least_removed_pct": deletion["still_flood_after_least_attended_deleted_pct"],
        },
    }


def build_audit(split_audit, rejected_audit, aider_audit):
    """The dataset-validation numbers: can a content-free model separate the classes?

    A balanced two-class problem scores 50% balanced accuracy by chance, so 50 is the
    reference line every bar is read against.
    """
    def summarise(audit):
        return {
            "target": audit["target"],
            "n_images": audit["n_images"],
            "size_leak_balanced_accuracy": audit["size_leak"]["balanced_accuracy"],
            "size_leak_plain_accuracy": audit["size_leak"]["plain_accuracy"],
            "single_class_size_block_pct": audit["size_leak"]["block_pct"],
            "source_leak_balanced_accuracy": audit["source_leak"]["balanced_accuracy"],
            "chance_baseline": 50.0,
            "verdict": audit["verdict"],
            "issues": [{"severity": i["severity"], "check": i["check"], "message": i["message"]}
                       for i in audit.get("issues", [])],
        }

    return {"split": summarise(split_audit),
            "rejected": summarise(rejected_audit),
            "aider": summarise(aider_audit),
            "note": audit_note(split_audit, rejected_audit, aider_audit)}


def audit_note(split_audit, rejected_audit, aider_audit):
    """One sentence stating what the three bars actually show, from their own numbers."""
    return (f"Dimensions alone reach {aider_audit['size_leak']['balanced_accuracy']:.0f}% balanced "
            f"accuracy on the class-balanced AIDER sample and "
            f"{rejected_audit['size_leak']['balanced_accuracy']:.0f}% on the rejected dataset, "
            f"against a {50:.0f}% chance baseline; the split actually trained on scores "
            f"{split_audit['size_leak']['balanced_accuracy']:.0f}%.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Collect reports/ into dashboard.json.")
    parser.add_argument("--report-dir", default=REPORT_DIR)
    parser.add_argument("--out", default=OUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    metrics = require(args.report_dir, "metrics.json")
    seed_summary = require(args.report_dir, "seed_summary.json")
    history = require(args.report_dir, "history.json")
    curves = require(args.report_dir, "curves.json")
    probabilities = require(args.report_dir, "test_probabilities.csv")
    val_sweep = require(args.report_dir, "threshold_sweep_val.csv")
    test_sweep = require(args.report_dir, "threshold_sweep_test.csv")
    benchmark = require(args.report_dir, "export_benchmark.json")
    gradcam = require(args.report_dir, "gradcam_stats.json")
    split_audit = require(args.report_dir, "audit_split.json")
    rejected_audit = require(args.report_dir, "audit_rejected.json")
    aider_audit = require(args.report_dir, "audit_aider.json")

    threshold = metrics["threshold"]["value"]

    # The deployed threshold is the single source of truth: every chart and caption in
    # the dashboard reads meta.threshold, so there is no second place to update.
    data = {
        "meta": build_meta(metrics, history, split_audit, threshold),
        "headline": {
            "recall": metrics["test_at_tuned_threshold"]["recall"],
            "precision": metrics["test_at_tuned_threshold"]["precision"],
            "f1": metrics["test_at_tuned_threshold"]["f1"],
            "accuracy": metrics["test_at_tuned_threshold"]["accuracy"],
            "roc_auc": metrics["test_roc_auc"],
            "average_precision": metrics["test_average_precision"],
            "false_negatives": len(metrics["false_negatives"]),
            "test_floods": metrics["test_floods"],
            "misclassified": metrics["misclassified_at_tuned_threshold"],
        },
        "comparison": build_comparison(seed_summary),
        "seed_note": seed_summary["std"] + "; " + seed_summary["threshold"],
        "seeds": seed_summary["seeds"],
        "history": build_history(history),
        "finetune_start_epoch": history["fine_tune_start_epoch"],
        "roc": [{"fpr": p["fpr"], "tpr": p["tpr"], "threshold": p["threshold"]} for p in curves["roc"]],
        "roc_auc": curves["roc_auc"],
        "pr": [{"recall": p["recall"], "precision": p["precision"], "threshold": p["threshold"]}
               for p in curves["pr"]],
        "average_precision": curves["average_precision"],
        "positive_share": curves["positive_share"],
        "operating_point": {
            "threshold": threshold,
            "recall": metrics["test_at_tuned_threshold"]["recall"],
            "precision": metrics["test_at_tuned_threshold"]["precision"],
            "fpr": 1.0 - metrics["test_at_tuned_threshold"]["specificity"],
        },
        "threshold_sweep": build_sweep(val_sweep, test_sweep),
        "confusion": build_confusion(metrics),
        "probabilities": build_probabilities(probabilities),
        "export": build_export(benchmark),
        "gradcam": build_gradcam(gradcam),
        "audit": build_audit(split_audit, rejected_audit, aider_audit),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(data, handle, indent=2)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"[export] Wrote {args.out}")
    print(f"[export] {size_kb:.1f} KB - threshold {threshold}, seed {data['meta']['seed_deployed']}, "
          f"{len(data['probabilities'])} test probabilities, {len(data['history'])} history rows, "
          f"{len(data['roc'])} ROC points, {len(data['pr'])} PR points")


if __name__ == "__main__":
    main()
