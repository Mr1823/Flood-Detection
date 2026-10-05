"""
train.py - train the two required variants and compare them.

  Variant A  "Without fine-tuning"  MobileNetV2 frozen; only the new head learns
                                    (up to EPOCHS_FROZEN epochs at LR_FROZEN).
  Variant B  "With fine-tuning"     continues from A's trained weights: MobileNetV2
                                    is unfrozen from 'block_13_expand' onward
                                    (BatchNorm stays frozen), the model is RECOMPILED
                                    with the 100x smaller LR_FINETUNE, then trained more.

Seeds: everything is repeated with N seeds (--seeds N, default 3: SEED, SEED+1,
SEED+2). With a test set of a few hundred images one run's number is noisy, so
results are reported as mean +/- std across seeds. Each seed trains both
variants into models/seed_<n>/ and reports/seed_<n>/.

Choosing the deployed model, using VALIDATION data only - the test set is scored
for the report and never influences any choice:
  the variant   = the one with the higher MEAN validation AUC across seeds
  the weights   = that variant's run with the highest validation AUC
                  -> models/best_model.keras (its curves and history are copied
                  to reports/training_curves.png and reports/history.json)

The dataset audit is a reporting step: if reports/audit_split.json exists its
numbers are printed at startup and stored in history.json. It never blocks.

Run from the project root:
  python src/train.py                      # 3 seeds
  python src/train.py --seeds 1 --epochs-frozen 1 --epochs-finetune 1   # quick check
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

import config  # prints the TensorFlow version and devices; imported before TensorFlow
import tensorflow as tf
from audit_dataset import summarise_audit
from dataset import (build_augmentation, check_labels_match_folders, compute_class_weights,
                     describe, make_datasets, save_sample_batch, spread_sample)
from model import (build_model, check_device_consistency, compile_model, count_trainable_params,
                   load_trained_model,
                   unfreeze_from)

METRICS = ("loss", "accuracy", "precision", "recall", "auc")
RUN_FILES = (config.HISTORY_FILE, config.TRAINING_CURVES_FILE,
             "training_log_frozen.csv", "training_log_finetuned.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 frozen, then fine-tuned, and keep the better variant.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--report-dir", default=config.REPORT_DIR)
    parser.add_argument("--epochs-frozen", type=int, default=config.EPOCHS_FROZEN)
    parser.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE)
    parser.add_argument("--lr-frozen", type=float, default=config.LR_FROZEN)
    parser.add_argument("--lr-finetune", type=float, default=config.LR_FINETUNE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--unfreeze-from", default=config.UNFREEZE_FROM)
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS,
                        help="number of seeds (SEED, SEED+1, ...); results are mean +/- std")
    parser.add_argument("--audit-report",
                        default=os.path.join(config.REPORT_DIR, config.AUDIT_SPLIT_REPORT))
    parser.add_argument("--verbose", type=int, default=2, choices=(0, 1, 2),
                        help="Keras output: 1 = progress bar, 2 = one line per epoch")
    args = parser.parse_args()
    if args.epochs_frozen < 1 or args.epochs_finetune < 1:
        parser.error("both phases need at least one epoch")
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    return args


def make_callbacks(variant, model_dir, report_dir, best_so_far=None):
    """The four callbacks both phases use. Returns (early_stopping, all_callbacks)."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=config.EARLY_STOP_PATIENCE,
        restore_best_weights=True, verbose=1)
    callbacks = [
        early_stopping,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.LR_PLATEAU_FACTOR,
            patience=config.LR_PLATEAU_PATIENCE, min_lr=config.MIN_LR, verbose=1),
        # Safety net during training: the file always holds the best epoch so far. In
        # phase 2 it starts from phase 1's best score, so a worse fine-tuning epoch can
        # never overwrite a better frozen one.
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, config.BEST_MODEL_FILE), monitor="val_auc", mode="max",
            save_best_only=True, initial_value_threshold=best_so_far, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(report_dir, f"training_log_{variant}.csv")),
    ]
    return early_stopping, callbacks


def fit_phase(model, variant, data, epochs, initial_epoch, args, model_dir, report_dir,
              best_so_far=None):
    """Run one training phase and leave the model holding that phase's best weights."""
    early_stopping, callbacks = make_callbacks(variant, model_dir, report_dir, best_so_far)
    start = time.time()
    history = model.fit(
        data["train"],
        validation_data=data["val"],
        epochs=initial_epoch + epochs,    # Keras counts epochs absolutely, so phase 2
        initial_epoch=initial_epoch,      # carries on from phase 1's numbering
        class_weight=data["class_weight"],
        callbacks=callbacks,
        shuffle=False,                    # the tf.data pipeline already reshuffles train every
                                          # epoch (dataset.py); fit's flag would be ignored anyway
        verbose=args.verbose,
    )
    # Keras 3 restores the best epoch at the end of fit(); older Keras only did so when
    # training actually stopped early. Restoring here makes the behaviour certain.
    if early_stopping.best_weights is not None:
        model.set_weights(early_stopping.best_weights)
    info = {
        "epochs_run": len(history.epoch),
        "best_epoch": int(early_stopping.best_epoch) + 1,      # 1-based, like the logs
        "stopped_early": bool(early_stopping.stopped_epoch > 0),
        "train_seconds": round(time.time() - start, 1),
    }
    return history, info


def score(model, data):
    """Loss and metrics on val and test (accuracy/precision/recall at threshold 0.5)."""
    results = {}
    for split in ("val", "test"):
        logs = model.evaluate(data[split], return_dict=True, verbose=0)
        results.update({f"{split}_{name}": float(logs[name]) for name in METRICS})
    return results


def select_variant(results):
    """Higher validation AUC wins; an exact tie goes to the lower validation loss."""
    a, b = results["frozen"], results["finetuned"]
    if a["val_auc"] != b["val_auc"]:
        return "finetuned" if b["val_auc"] > a["val_auc"] else "frozen"
    return "finetuned" if b["val_loss"] < a["val_loss"] else "frozen"


def check_saved_model(path, val_ds, expected_auc):
    """Reload the saved file and confirm it reproduces the validation AUC it was chosen for."""
    reloaded = load_trained_model(path)
    check_device_consistency(reloaded, spread_sample(val_ds))   # the saved model, GPU vs CPU
    auc = reloaded.evaluate(val_ds, return_dict=True, verbose=0)["auc"]
    status = "matches" if abs(auc - expected_auc) < 1e-4 else f"WARNING - expected {expected_auc:.4f}"
    print(f"[train] Reloaded {config.rel(path)}: validation AUC {auc:.4f} ({status})")


def combine_histories(phases):
    """Join the per-epoch logs of both phases into one record, labelling each epoch's phase."""
    combined = {"epoch": [], "phase": []}
    for variant, history in phases:
        combined["epoch"] += [e + 1 for e in history.epoch]          # 1-based for people
        combined["phase"] += [variant] * len(history.epoch)
    for key in METRICS + tuple(f"val_{m}" for m in METRICS) + ("learning_rate",):
        combined[key] = [float(v) for _, history in phases for v in history.history.get(key, [])]
    return combined


def plot_training_curves(history, fine_tune_start, out_path):
    """Accuracy, loss and AUC across both phases; a dashed line marks where fine-tuning began."""
    plt = config.use_report_style()
    from matplotlib.ticker import MaxNLocator

    epochs = history["epoch"]
    panels = (("accuracy", "Accuracy"), ("loss", "Loss (binary cross-entropy)"), ("auc", "ROC AUC"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, title) in zip(axes, panels):
        ax.plot(epochs, history[key], color=config.PLOT_SERIES_1, marker="o", markersize=4,
                label="train")
        ax.plot(epochs, history[f"val_{key}"], color=config.PLOT_SERIES_2, marker="o",
                markersize=4, label="validation")
        ax.axvline(fine_tune_start - 0.5, color=config.PLOT_INK_2, linestyle="--", linewidth=1.2)
        ax.annotate("fine-tuning starts", xy=(fine_tune_start - 0.5, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(5, -6),
                    textcoords="offset points", ha="left", va="top", fontsize=9,
                    color=config.PLOT_INK_2,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": config.PLOT_SURFACE,
                          "edgecolor": "none", "alpha": 0.9})   # stays readable over a curve
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Training history: frozen backbone, then fine-tuning",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.0, "Train metrics are measured with augmentation, dropout and class weights "
             "switched on, so they can sit below the validation curve.",
             fontsize=8.5, color=config.PLOT_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, dpi=config.PLOT_DPI, bbox_inches="tight")   # keeps the footnote in frame
    plt.close(fig)


def print_comparison(results, selected):
    """The required comparison table, filled from the measured values."""
    print("\n================ MODEL COMPARISON ================")
    print(f"{'Model':<25}{'Val loss':<11}{'Val acc':<10}Test acc")
    for variant in ("frozen", "finetuned"):
        r = results[variant]
        print(f"{config.VARIANT_LABELS[variant]:<25}{r['val_loss']:<11.4f}"
              f"{r['val_accuracy']:<10.4f}{r['test_accuracy']:.4f}")
    print()
    print(f"Selected model: {config.VARIANT_LABELS[selected]}")


def load_data(seed, args, inspect):
    """Pipelines for one seed (the seed sets the training shuffle order)."""
    train_ds, val_ds, test_ds, class_names = make_datasets(
        args.data_dir, config.IMG_SIZE, args.batch_size, seed, validate=inspect)
    if inspect:                       # data checks are the same for every seed: run them once
        describe(train_ds, class_names, args.data_dir)
        check_labels_match_folders(val_ds, "val", args.data_dir)
        check_labels_match_folders(test_ds, "test", args.data_dir)
    class_weight = compute_class_weights(os.path.join(args.data_dir, "train"))
    print("[train] Class weights: "
          + ", ".join(f"{class_names[i]} ({i}) = {w:.3f}" for i, w in class_weight.items()))
    return {"train": train_ds, "val": val_ds, "test": test_ds, "class_names": class_names,
            "class_weight": class_weight}


def run_seed(seed, args, model_dir, report_dir, audit, inspect):
    """Train and compare both variants with one seed; returns the run's record."""
    tf.keras.backend.clear_session()  # a fresh Keras state for every seed
    config.set_seeds(seed)
    config.ensure_dirs(model_dir, report_dir)
    data = load_data(seed, args, inspect)
    if inspect:
        sample_path = os.path.join(args.report_dir, config.SAMPLE_BATCH_FILE)
        save_sample_batch(data["train"], data["class_names"], sample_path,
                          augmentation=build_augmentation())
        print(f"[train] Saved {config.rel(sample_path)} - check the labels before trusting any result")

    # ------------------------------------------- Variant A: frozen backbone
    model = build_model(augmentation=build_augmentation())
    compile_model(model, args.lr_frozen)
    # Before spending any time on training: the GPU must compute what the CPU computes.
    check_device_consistency(model, spread_sample(data["val"]))
    print(f"\n[model] {len(model.layers)} layers, {model.count_params():,} parameters, "
          f"{count_trainable_params(model):,} trainable (the new head only); "
          f"'{config.GRADCAM_LAYER}' is reachable for Grad-CAM")
    print(f"\n=== Seed {seed} - Variant A - frozen backbone: up to {args.epochs_frozen} epochs, "
          f"LR {args.lr_frozen:g} ===")
    hist_a, info_a = fit_phase(model, "frozen", data, args.epochs_frozen, 0, args,
                               model_dir, report_dir)
    results = {"frozen": {**score(model, data), **info_a}}
    model.save(os.path.join(model_dir, config.VARIANT_MODEL_FILES["frozen"]))

    # ------------------------------- Variant B: A's weights, then fine-tune
    print(f"\n=== Seed {seed} - Variant B - fine-tuning from '{args.unfreeze_from}': up to "
          f"{args.epochs_finetune} more epochs, LR {args.lr_finetune:g} ===")
    unfreeze_from(model, args.unfreeze_from)
    compile_model(model, args.lr_finetune)   # mandatory: Keras reads the trainable flags at compile time
    hist_b, info_b = fit_phase(model, "finetuned", data, args.epochs_finetune, len(hist_a.epoch),
                               args, model_dir, report_dir,
                               best_so_far=results["frozen"]["val_auc"])
    results["finetuned"] = {**score(model, data), **info_b}
    model.save(os.path.join(model_dir, config.VARIANT_MODEL_FILES["finetuned"]))

    # ----------------------------------------------------- compare + select
    selected = select_variant(results)
    best_path = os.path.join(model_dir, config.BEST_MODEL_FILE)
    shutil.copyfile(os.path.join(model_dir, config.VARIANT_MODEL_FILES[selected]), best_path)
    check_saved_model(best_path, data["val"], results[selected]["val_auc"])

    # ------------------------------------------------- history and curves
    history = combine_histories([("frozen", hist_a), ("finetuned", hist_b)])
    fine_tune_start = len(hist_a.epoch) + 1
    record = {
        "seed": seed,
        "audit": audit,
        "fine_tune_start_epoch": fine_tune_start,
        "history": history,
        "variants": results,
        "selected": selected,
        "selection_rule": "highest validation AUC (exact tie: lower validation loss); "
                          "the test set is never used to choose",
        "metric_notes": "accuracy/precision/recall use threshold 0.5; AUC is Keras' "
                        "200-threshold approximation",
        "hyperparameters": {
            "img_size": list(config.IMG_SIZE), "batch_size": args.batch_size,
            "epochs_frozen": args.epochs_frozen, "lr_frozen": args.lr_frozen,
            "epochs_finetune": args.epochs_finetune, "lr_finetune": args.lr_finetune,
            "unfreeze_from": args.unfreeze_from, "dropout_1": config.DROPOUT_1,
            "dense_units": config.DENSE_UNITS, "dropout_2": config.DROPOUT_2,
            "class_weight": data["class_weight"], "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "lr_plateau": [config.LR_PLATEAU_FACTOR, config.LR_PLATEAU_PATIENCE, config.MIN_LR],
            "training_crop": {"area": list(config.CROP_SCALE), "aspect": list(config.CROP_ASPECT)},
            "augmentation": {"flip": config.AUG_FLIP, "rotation": config.AUG_ROTATION,
                             "brightness": config.AUG_BRIGHTNESS, "contrast": config.AUG_CONTRAST},
        },
    }
    with open(os.path.join(report_dir, config.HISTORY_FILE), "w") as handle:
        json.dump(record, handle, indent=2)
    plot_training_curves(history, fine_tune_start,
                         os.path.join(report_dir, config.TRAINING_CURVES_FILE))
    print(f"[train] Saved {config.rel(best_path)} and the history, curves and logs in "
          f"{config.rel(report_dir)}/")

    print(f"[train] Validation AUC: {results['frozen']['val_auc']:.4f} without vs "
          f"{results['finetuned']['val_auc']:.4f} with fine-tuning "
          "(accuracies below use the default 0.5 threshold)")
    print_comparison(results, selected)
    return {"seed": seed, "variants": results, "selected": selected,
            "model_dir": model_dir, "report_dir": report_dir}


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0   # sample std across seeds
    return {"mean": float(values.mean()), "std": std, "values": values.tolist()}


def summarise_seeds(runs, args):
    """Mean +/- std across seeds, and the deployed model - chosen on validation data only."""
    keys = ("val_loss", "val_accuracy", "val_auc",
            "test_accuracy", "test_auc", "test_precision", "test_recall")
    stats = {v: {k: _mean_std([r["variants"][v][k] for r in runs]) for k in keys}
             for v in ("frozen", "finetuned")}

    # Variant: higher MEAN validation AUC (exact tie: lower mean validation loss).
    selected = max(("frozen", "finetuned"),
                   key=lambda v: (stats[v]["val_auc"]["mean"], -stats[v]["val_loss"]["mean"]))
    # Weights: that variant's single run with the highest validation AUC.
    best = max(runs, key=lambda r: r["variants"][selected]["val_auc"])
    shutil.copyfile(os.path.join(best["model_dir"], config.VARIANT_MODEL_FILES[selected]),
                    os.path.join(args.model_dir, config.BEST_MODEL_FILE))
    for name in RUN_FILES:            # top-level reports/ always describe the deployed run
        shutil.copyfile(os.path.join(best["report_dir"], name), os.path.join(args.report_dir, name))
    history_path = os.path.join(args.report_dir, config.HISTORY_FILE)
    with open(history_path) as handle:
        record = json.load(handle)
    record["deployed"] = {"variant": selected, "seed": best["seed"],
                          "rule": "variant with the higher mean validation AUC across seeds; "
                                  "weights from its run with the highest validation AUC"}
    with open(history_path, "w") as handle:
        json.dump(record, handle, indent=2)

    summary = {"seeds": [r["seed"] for r in runs], "selected_variant": selected,
               "deployed_seed": best["seed"], "per_seed_choice": {r["seed"]: r["selected"] for r in runs},
               "std": "sample standard deviation (ddof=1) across seeds",
               "threshold": "accuracy, precision and recall use the default 0.5 threshold",
               "variants": stats}
    with open(os.path.join(args.report_dir, config.SEED_SUMMARY_FILE), "w") as handle:
        json.dump(summary, handle, indent=2)

    def cell(variant, key, width):
        s = stats[variant][key]
        return f"{s['mean']:.4f} ± {s['std']:.4f}".ljust(width)

    n = len(runs)
    print(f"\n================ MODEL COMPARISON ({n} seeds, mean ± std) ================")
    print(f"{'Model':<25}{'Val loss':<20}{'Val acc':<20}Test acc")
    for v in ("frozen", "finetuned"):
        print(f"{config.VARIANT_LABELS[v]:<25}{cell(v, 'val_loss', 20)}{cell(v, 'val_accuracy', 20)}"
              f"{cell(v, 'test_accuracy', 0)}")
    print(f"\n{'Test set (threshold 0.5)':<25}{'Accuracy':<20}{'AUC':<20}{'Precision':<20}Recall")
    for v in ("frozen", "finetuned"):
        print(f"{config.VARIANT_LABELS[v]:<25}{cell(v, 'test_accuracy', 20)}{cell(v, 'test_auc', 20)}"
              f"{cell(v, 'test_precision', 20)}{cell(v, 'test_recall', 0)}")
    print(f"\nValidation AUC: {cell('frozen', 'val_auc', 0)} without vs "
          f"{cell('finetuned', 'val_auc', 0)} with fine-tuning")
    print(f"\nSelected model: {config.VARIANT_LABELS[selected]}")
    print(f"  deployed weights: seed {best['seed']} (its highest validation AUC, "
          f"{best['variants'][selected]['val_auc']:.4f}) -> {config.rel(os.path.join(args.model_dir, config.BEST_MODEL_FILE))}")


def main():
    args = parse_args()
    config.set_seeds(config.SEED)
    config.ensure_dirs(args.model_dir, args.report_dir)

    # The dataset audit is reported, never enforced.
    audit = summarise_audit(args.data_dir, args.audit_report)

    seeds = [config.SEED + k for k in range(args.seeds)]
    multi = len(seeds) > 1
    runs = []
    for k, seed in enumerate(seeds):
        model_dir = os.path.join(args.model_dir, f"seed_{seed}") if multi else args.model_dir
        report_dir = os.path.join(args.report_dir, f"seed_{seed}") if multi else args.report_dir
        runs.append(run_seed(seed, args, model_dir, report_dir, audit, inspect=(k == 0)))
    if multi:
        summarise_seeds(runs, args)


if __name__ == "__main__":
    main()
