"""
evaluate.py - evaluate models/best_model.keras and choose the decision threshold.

THRESHOLD SELECTION - on VALIDATION data only, then reported on test
    A missed flood costs lives; a false alarm costs an inspection. So the rule
    is: the LOWEST threshold whose validation precision is still >= MIN_PRECISION
    (0.85). Lowering the threshold can only raise recall, so this maximises
    recall subject to the precision floor. The threshold is chosen on the
    validation set and then applied, unchanged, to the test set: choosing it on
    test and reporting test would be selection on the test set, and the test
    numbers would be optimistic. Both sweeps are saved.

Outputs
  models/threshold.json                {"threshold": x, "rationale": "...", ...}
  reports/threshold_sweep_val.csv      the sweep the threshold is chosen from
  reports/threshold_sweep_test.csv     the same sweep on test, for reporting only
  reports/threshold_sweep.png          precision / recall / F1 vs threshold, both splits
  reports/confusion_matrix.png         test set at 0.5 and at the tuned threshold
  reports/roc_curve.png, reports/pr_curve.png
  reports/misclassified.png            up to 16 wrong test predictions
  reports/results_dashboard.png        training curves, confusion matrix, ROC, PR
  reports/metrics.json                 every number (the README is filled from it)
  reports/false_negatives.csv          the floods the system would miss

    python src/evaluate.py
"""
import argparse
import csv
import json
import os

import numpy as np

import config  # prints the TensorFlow version and devices; imported before TensorFlow
from dataset import load_for_model, make_datasets, spread_sample
from model import check_device_consistency, load_trained_model


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def predict(model, ds):
    """Flood probabilities and true labels, in the dataset's (= file) order."""
    probs = model.predict(ds, verbose=0).ravel()
    labels = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    return probs, labels


def confusion(labels, probs, threshold):
    """[[TN, FP], [FN, TP]]: rows = actual (no_flood, flood), columns = predicted."""
    pred = (probs >= threshold).astype(int)
    tn = int(((labels == 0) & (pred == 0)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    tp = int(((labels == 1) & (pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def scores(labels, probs, threshold):
    """Every threshold-dependent metric at one threshold (flood = positive class)."""
    (tn, fp), (fn, tp) = confusion(labels, probs, threshold).tolist()   # plain ints (JSON-safe)
    precision = tp / (tp + fp) if tp + fp else 0.0        # no flood predicted at all -> 0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": round(float(threshold), 2), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "specificity": specificity,
            "accuracy": (tp + tn) / len(labels), "balanced_accuracy": (recall + specificity) / 2}


def sweep(labels, probs):
    thresholds = np.arange(config.THRESHOLD_START, config.THRESHOLD_STOP + 1e-9, config.THRESHOLD_STEP)
    return [scores(labels, probs, t) for t in thresholds]


def choose_threshold(val_sweep):
    """Lowest threshold whose VALIDATION precision >= MIN_PRECISION, i.e. maximum recall
    under the precision floor. Falls back to the highest-precision threshold if none qualifies."""
    ok = [row for row in val_sweep if row["precision"] >= config.MIN_PRECISION]
    if ok:
        return ok[0], True
    return max(val_sweep, key=lambda row: (row["precision"], -row["threshold"])), False


def threshold_for_recall(val_sweep, target):
    """Highest threshold whose validation recall still reaches `target` (best precision there)."""
    ok = [row for row in val_sweep if row["recall"] >= target]
    return ok[-1] if ok else None


def write_sweep(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_confusion(title, matrix):
    """Rows = actual, columns = predicted, order no_flood then flood - labelled so it cannot be misread."""
    print(f"\n{title}")
    print(f"{'':>20}{'predicted no_flood':>20}{'predicted flood':>18}")
    for name, row in zip(config.CLASS_NAMES, matrix):
        print(f"{'actual ' + name:>20}{row[0]:>20}{row[1]:>18}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_sweeps(val_rows, test_rows, chosen, out_path):
    plt = config.use_report_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, rows, title in ((axes[0], val_rows, "Validation - the threshold is chosen here"),
                            (axes[1], test_rows, "Test - reported only, never used to choose")):
        t = [r["threshold"] for r in rows]
        for key, colour, label in (("precision", config.PLOT_SERIES_1, "precision"),
                                   ("recall", config.PLOT_SERIES_2, "recall"),
                                   ("f1", config.PLOT_INK_2, "F1")):
            ax.plot(t, [r[key] for r in rows], color=colour, marker="o", markersize=4, label=label)
        ax.axvline(chosen["threshold"], color=config.PLOT_INK, linewidth=1.2, linestyle="--")
        ax.axhline(config.MIN_PRECISION, color=config.PLOT_MUTED, linewidth=1)
        ax.annotate(f"precision floor {config.MIN_PRECISION}", xy=(0.02, config.MIN_PRECISION),
                    xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                    fontsize=8.5, color=config.PLOT_MUTED)
        ax.annotate(f"chosen {chosen['threshold']:.2f}", xy=(chosen["threshold"], 0.02),
                    xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset points",
                    fontsize=9, color=config.PLOT_INK)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Decision threshold (flood if probability >= threshold)")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.PLOT_DPI)
    plt.close(fig)


def draw_confusion(ax, matrix, title):
    import seaborn as sns

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, square=True,
                linewidths=2, linecolor=config.PLOT_SURFACE, annot_kws={"fontsize": 14},
                xticklabels=[f"pred. {c}" for c in config.CLASS_NAMES],
                yticklabels=[f"actual {c}" for c in config.CLASS_NAMES])
    ax.set_title(title, loc="left")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="y", rotation=0)


def plot_confusions(m_default, m_tuned, threshold, out_path):
    plt = config.use_report_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    draw_confusion(axes[0], m_default, "Test set, threshold 0.50")
    draw_confusion(axes[1], m_tuned, f"Test set, tuned threshold {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.PLOT_DPI)
    plt.close(fig)


def draw_roc(ax, labels, probs, chosen_row):
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    ax.plot(fpr, tpr, color=config.PLOT_SERIES_1, label=f"MobileNetV2 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=config.PLOT_MUTED, linewidth=1, label="chance (AUC = 0.5)")
    ax.plot(1 - chosen_row["specificity"], chosen_row["recall"], "o", color=config.PLOT_INK,
            markersize=8, label=f"tuned threshold {chosen_row['threshold']:.2f}")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve - test set", loc="left")
    ax.legend(loc="lower right")
    return auc


def draw_pr(ax, labels, probs, chosen_row):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    base = labels.mean()
    ax.plot(recall, precision, color=config.PLOT_SERIES_1, label=f"MobileNetV2 (AP = {ap:.3f})")
    ax.axhline(base, color=config.PLOT_MUTED, linewidth=1, label=f"chance (flood share {base:.2f})")
    ax.plot(chosen_row["recall"], chosen_row["precision"], "o", color=config.PLOT_INK, markersize=8,
            label=f"tuned threshold {chosen_row['threshold']:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curve - test set", loc="left")
    ax.legend(loc="lower left")
    return ap


def plot_single(draw, labels, probs, chosen_row, out_path):
    plt = config.use_report_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    value = draw(ax, labels, probs, chosen_row)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.PLOT_DPI)
    plt.close(fig)
    return value


def plot_misclassified(paths, labels, probs, threshold, out_path, limit=16):
    plt = config.use_report_style()
    wrong = [i for i in range(len(labels)) if int(probs[i] >= threshold) != labels[i]]
    if not wrong:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No misclassified test images at threshold {threshold:.2f}.",
                ha="center", va="center", fontsize=13)
        fig.savefig(out_path, dpi=config.PLOT_DPI)
        plt.close(fig)
        return 0
    shown = wrong[:limit]
    cols = 4
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 4.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, shown):
        display, _ = load_for_model(paths[i])
        ax.imshow(display)
        true, pred = config.CLASS_NAMES[labels[i]], config.CLASS_NAMES[int(probs[i] >= threshold)]
        ax.set_title(f"{true} -> {pred}\nP(flood) = {probs[i]:.3f}\n{os.path.basename(paths[i])}",
                     fontsize=9)
    fig.suptitle(f"Misclassified test images at threshold {threshold:.2f} "
                 f"({len(shown)} of {len(wrong)} shown) - true -> predicted",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)   # room for the three-line captions
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return len(wrong)


def plot_dashboard(history_path, m_tuned, threshold, labels, probs, chosen_row, out_path):
    plt = config.use_report_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    ax = axes[0, 0]
    if os.path.isfile(history_path):
        with open(history_path) as handle:
            record = json.load(handle)
        h = record["history"]
        ax.plot(h["epoch"], h["accuracy"], color=config.PLOT_SERIES_1, marker="o", markersize=3,
                label="train")
        ax.plot(h["epoch"], h["val_accuracy"], color=config.PLOT_SERIES_2, marker="o", markersize=3,
                label="validation")
        ax.axvline(record["fine_tune_start_epoch"] - 0.5, color=config.PLOT_INK_2, linestyle="--",
                   linewidth=1.2)
        ax.text(record["fine_tune_start_epoch"] - 0.4, 0.97, "fine-tuning starts", fontsize=9,
                va="top", color=config.PLOT_INK_2, transform=ax.get_xaxis_transform(),
                bbox={"boxstyle": "round,pad=0.2", "facecolor": config.PLOT_SURFACE,
                      "edgecolor": "none", "alpha": 0.9})
        ax.set_xlabel("Epoch")
        ax.set_title("Training accuracy (deployed run)", loc="left")
        ax.legend(loc="lower right")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "history.json not found", ha="center")
    draw_confusion(axes[0, 1], m_tuned, f"Test set, tuned threshold {threshold:.2f}")
    draw_roc(axes[1, 0], labels, probs, chosen_row)
    draw_pr(axes[1, 1], labels, probs, chosen_row)
    fig.suptitle("Flood detection - results dashboard", x=0.01, ha="left", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=config.PLOT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model and choose the decision threshold.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--report-dir", default=config.REPORT_DIR)
    parser.add_argument("--model", default=None, help="model file (default: <model-dir>/best_model.keras)")
    parser.add_argument("--min-precision", type=float, default=config.MIN_PRECISION)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main():
    from sklearn.metrics import classification_report

    args = parse_args()
    config.set_seeds(args.seed)
    config.MIN_PRECISION = args.min_precision            # one value used everywhere below
    config.ensure_dirs(args.model_dir, args.report_dir)
    model_path = args.model or os.path.join(args.model_dir, config.BEST_MODEL_FILE)
    model = load_trained_model(model_path, compile=False)
    _, val_ds, test_ds, _ = make_datasets(args.data_dir, config.IMG_SIZE, config.BATCH_SIZE,
                                          args.seed, validate=False)
    check_device_consistency(model, spread_sample(val_ds))   # numbers below come from the GPU
    val_probs, val_labels = predict(model, val_ds)
    test_probs, test_labels = predict(model, test_ds)
    test_paths = test_ds.file_paths

    # (a) Threshold: swept and chosen on VALIDATION, then applied to test.
    val_rows, test_rows = sweep(val_labels, val_probs), sweep(test_labels, test_probs)
    write_sweep(os.path.join(args.report_dir, "threshold_sweep_val.csv"), val_rows)
    write_sweep(os.path.join(args.report_dir, "threshold_sweep_test.csv"), test_rows)
    chosen, met_floor = choose_threshold(val_rows)
    threshold = chosen["threshold"]
    test_at = scores(test_labels, test_probs, threshold)
    test_default = scores(test_labels, test_probs, config.DEFAULT_THRESHOLD)
    recall_row = threshold_for_recall(val_rows, config.TARGET_RECALL)
    recall_test = scores(test_labels, test_probs, recall_row["threshold"]) if recall_row else None

    rationale = (f"Lowest threshold whose VALIDATION precision is still >= {config.MIN_PRECISION} "
                 f"(validation precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f}). "
                 "A missed flood costs lives while a false alarm costs an inspection, so recall is "
                 "maximised subject to the precision floor. Chosen on validation data only.")
    if not met_floor:
        rationale = (f"No threshold reached validation precision {config.MIN_PRECISION}; the "
                     f"highest-precision threshold was used instead. " + rationale)
    with open(os.path.join(args.model_dir, config.THRESHOLD_FILE), "w") as handle:
        json.dump({"threshold": threshold, "rationale": rationale, "selected_on": "validation",
                   "min_precision": config.MIN_PRECISION, "validation": chosen}, handle, indent=2)
    plot_sweeps(val_rows, test_rows, chosen, os.path.join(args.report_dir, "threshold_sweep.png"))

    # (b) Printed results, in the required order.
    print(f"\nChosen flood threshold: {threshold:.2f}  (saved to {config.THRESHOLD_FILE})")
    print(f"  Why: {rationale}")
    if recall_row:
        print(f"  For comparison, recall >= {config.TARGET_RECALL} on validation needs threshold "
              f"{recall_row['threshold']:.2f}, at validation precision {recall_row['precision']:.3f} "
              f"(test: precision {recall_test['precision']:.3f}, recall {recall_test['recall']:.3f}).")
    else:
        print(f"  No swept threshold reaches recall {config.TARGET_RECALL} on validation.")
    m_default = confusion(test_labels, test_probs, config.DEFAULT_THRESHOLD)
    m_tuned = confusion(test_labels, test_probs, threshold)
    print_confusion("Confusion matrix with default 0.5 threshold (test set; rows = actual, "
                    "columns = predicted):", m_default)
    print_confusion(f"Confusion matrix with tuned threshold {threshold:.2f} (test set; rows = actual, "
                    "columns = predicted):", m_tuned)
    print("\nClassification report (test set, tuned threshold):")
    print(classification_report(test_labels, (test_probs >= threshold).astype(int),
                                target_names=config.CLASS_NAMES, digits=3, zero_division=0))

    # (c) Figures.
    roc_auc = plot_single(draw_roc, test_labels, test_probs, test_at,
                          os.path.join(args.report_dir, "roc_curve.png"))
    ap = plot_single(draw_pr, test_labels, test_probs, test_at,
                     os.path.join(args.report_dir, "pr_curve.png"))
    print(f"ROC AUC: {roc_auc:.4f}   (average precision {ap:.4f})")
    plot_confusions(m_default, m_tuned, threshold, os.path.join(args.report_dir, "confusion_matrix.png"))
    n_wrong = plot_misclassified(test_paths, test_labels, test_probs, threshold,
                                 os.path.join(args.report_dir, "misclassified.png"))
    plot_dashboard(os.path.join(args.report_dir, config.HISTORY_FILE), m_tuned, threshold,
                   test_labels, test_probs, test_at,
                   os.path.join(args.report_dir, "results_dashboard.png"))

    # (d) False negatives: the floods the system would miss.
    fn_rows = [{"file": config.rel(test_paths[i]), "p_flood": round(float(test_probs[i]), 4)}
               for i in range(len(test_labels))
               if test_labels[i] == 1 and test_probs[i] < threshold]
    with open(os.path.join(args.report_dir, "false_negatives.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "p_flood"])
        writer.writeheader()
        writer.writerows(fn_rows)
    print(f"\nFalse negatives at threshold {threshold:.2f} - floods the system would miss "
          f"({len(fn_rows)} of {int(test_labels.sum())} test floods):")
    for row in fn_rows or [{"file": "(none)", "p_flood": ""}]:
        print(f"  {row['file']}  {row['p_flood']}")

    # Every number, for the README.
    metrics = {
        "model": config.rel(model_path),
        "test_images": int(len(test_labels)), "test_floods": int(test_labels.sum()),
        "val_images": int(len(val_labels)), "val_floods": int(val_labels.sum()),
        "threshold": {"value": threshold, "selected_on": "validation", "rationale": rationale,
                      "min_precision": config.MIN_PRECISION, "validation_at_threshold": chosen,
                      "target_recall": config.TARGET_RECALL,
                      "target_recall_threshold_val": recall_row,
                      "target_recall_threshold_test": recall_test},
        "test_at_default_0.5": test_default,
        "test_at_tuned_threshold": test_at,
        "test_roc_auc": float(roc_auc), "test_average_precision": float(ap),
        "misclassified_at_tuned_threshold": n_wrong,
        "false_negatives": fn_rows,
        "confusion_matrix_note": "rows = actual, columns = predicted, order no_flood then flood",
    }
    for name in (config.SEED_SUMMARY_FILE, config.HISTORY_FILE):
        path = os.path.join(args.report_dir, name)
        if os.path.isfile(path):
            with open(path) as handle:
                data = json.load(handle)
            metrics[os.path.splitext(name)[0]] = data if name == config.SEED_SUMMARY_FILE else {
                k: data[k] for k in ("seed", "variants", "selected", "deployed", "audit") if k in data}
    with open(os.path.join(args.report_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\n[evaluate] Saved metrics.json, threshold sweeps, figures and false_negatives.csv in "
          f"{config.rel(args.report_dir)}/ and {config.THRESHOLD_FILE} in {config.rel(args.model_dir)}/")


if __name__ == "__main__":
    main()
