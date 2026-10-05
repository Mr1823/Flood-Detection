"""
config.py - every constant of the Flood Detection System lives here.

Other modules do `import config` and read values such as config.IMG_SIZE, so a
hyperparameter is changed in exactly one place and no unexplained "magic
numbers" appear anywhere else in the project.

Importing this module also imports TensorFlow and prints its version and the
devices it can see, so every run shows whether the Apple-Silicon (Metal) GPU
is actually being used.
"""
import os
import random

import numpy as np

# Hide TensorFlow's C++ INFO chatter; warnings and errors are still shown.
# It must be set before TensorFlow is imported to have any effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf  # noqa: E402  (deliberately imported after the env var)

# tensorflow-metal 1.2.0 registers its own graph optimiser, and on this setup it
# corrupts the Dense(128, relu) layer in GRAPH mode - the mode fit(), evaluate()
# and predict() use: the same model gave P(flood) 0.258 on the Metal GPU and 0.565
# on the CPU for one image. Found by comparing every layer, GPU graph vs CPU: all
# agreed to ~1e-6 up to global pooling, then the Dense output was off by 160 %.
# Switching off ONLY the plugin's optimiser fixes it (difference back to ~1e-6)
# while TensorFlow's own graph optimisations and the Metal GPU stay in use.
# model.check_device_consistency() re-checks this before training and evaluation.
tf.config.optimizer.set_experimental_options({"use_plugin_optimizers": False})

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)            # MobileNetV2's native ImageNet resolution
BATCH_SIZE = 32
CHANNELS = 3                     # RGB
CLASS_NAMES = ["no_flood", "flood"]      # index 1 = positive class = flood

# Paths are anchored to the project folder (the parent of src/), so scripts
# find data/, models/ and reports/ even if launched from another directory.
# backend/src/config.py -> backend/src -> backend -> the project root, where the
# dataset, models and reports live as siblings of backend/ and frontend/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
REJECTED_SUBDIR = "rejected"     # data/rejected/: the first dataset, kept as audit evidence only

# Source dataset: AIDER (Kyrkou, Zenodo 3888300), unzipped into aider_raw/.
# Only two of its five class folders are used; the rest are ignored.
SOURCE_DIR = os.path.join(PROJECT_ROOT, "aider_raw", "AIDER")
SOURCE_CLASS_DIRS = {"flood": "flooded_areas", "no_flood": "normal"}
NO_FLOOD_SUBSAMPLE = 1000        # no_flood images kept after de-duplication (from 4,390)

SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)     # stratified train / val / test

# Near-duplicate detection (split_data.py). MD5 only catches byte-identical files;
# the same photo saved at another size or JPEG quality has different bytes. A
# difference hash on a 16x16 grid gives 256 bits per image. On the first (rejected)
# dataset, resized or re-cropped copies of the same picture differed in 0-16 bits,
# while the closest pair of genuinely different images (two app icons) differed in
# 33 - so the 25-bit cut-off sits inside that gap. A 64-bit hash
# with a 5-bit threshold was tried first and flagged different icons and scenes as
# copies: flat, low-detail images need the finer 256-bit grid.
NEAR_DUP_HASH_SIZE = 16
NEAR_DUP_MAX_BITS = 25           # <= 25 of 256 bits (10 %) = treated as the same photo
# A perfectly flat image (e.g. an all-black video frame in AIDER's normal class)
# shows no scene, and its difference hash is degenerate - it "matched" a calm-sea
# frame. Images whose pixel standard deviation is below this are skipped as blank.
BLANK_PIXEL_STD = 1.0

# The only file types TensorFlow's image decoder reads. split_data.py reports
# anything else instead of copying it.
IMAGE_EXTENSIONS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")

# ---------------------------------------------------------------------------
# Cropping - the mitigation for the shortcut the audit measured (dataset.py)
# Training: a random window covering 70-100 % of the image area with an aspect
# ratio between 3:4 and 4:3, resized to 224x224 - a different framing every epoch.
# Val / test / inference: the largest CENTRED window whose aspect ratio lies in
# that same 3:4-4:3 range (the whole image, if it already does).
# ---------------------------------------------------------------------------
CROP_SCALE = (0.7, 1.0)          # fraction of the image area a training crop covers
CROP_ASPECT = (3 / 4, 4 / 3)     # width / height range of every crop

# ---------------------------------------------------------------------------
# Augmentation (training only; these layers live inside the model graph)
# ---------------------------------------------------------------------------
AUG_FLIP = "horizontal"          # never vertical: upside-down floods are not a real input
AUG_ROTATION = 0.05              # fraction of a full turn: +/-0.05 x 360 deg = +/-18 deg.
                                 # (0.15 = +/-54 deg broke the horizon prior flood scenes have.)
                                 # No RandomZoom: the random crop above already varies scale.
AUG_BRIGHTNESS = 0.2             # shift brightness by up to 20 % of the 0..255 range
AUG_CONTRAST = 0.2               # scale contrast by a factor in [0.8, 1.2]
PIXEL_RANGE = (0.0, 255.0)       # raw pixel range the model receives (see dataset.py)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
DROPOUT_1 = 0.3
DENSE_UNITS = 128
DROPOUT_2 = 0.2
GRADCAM_LAYER = "Conv_1"         # last conv layer of MobileNetV2 (7x7x1280 feature maps)

# Phase 1 - frozen backbone
EPOCHS_FROZEN = 15
LR_FROZEN = 1e-3
# Phase 2 - fine-tuning
EPOCHS_FINETUNE = 15
LR_FINETUNE = 1e-5
UNFREEZE_FROM = "block_13_expand"        # layer name, not index

N_SEEDS = 3                      # train.py repeats everything with seeds SEED, SEED+1, ...

# Callbacks
EARLY_STOP_PATIENCE = 5         # epochs without a val_auc improvement before stopping
LR_PLATEAU_FACTOR = 0.5          # halve the learning rate ...
LR_PLATEAU_PATIENCE = 3          # ... after 3 epochs without a val_loss improvement
MIN_LR = 1e-7

# The two variants train.py compares, with the labels used in its table.
VARIANT_LABELS = {
    "frozen": "Without fine-tuning",
    "finetuned": "With fine-tuning",
}

# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
MIN_PRECISION = 0.85             # floor we refuse to go below
TARGET_RECALL = 0.95             # also report the threshold hitting this
DEFAULT_THRESHOLD = 0.5
THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP = 0.05, 0.95, 0.05

# ---------------------------------------------------------------------------
# Dataset audit (audit_dataset.py) - shortcut checks, reported before training.
# The audit REPORTS; it does not block training. Accuracies are BALANCED
# accuracies (mean of the per-class recalls), so chance is 50 % however lopsided
# the classes are. The size and duplicate cut-offs below are judgement calls.
# The source-leak test has no pass line: its score is reported as a lower bound.
# ---------------------------------------------------------------------------
AUDIT_SPLIT_REPORT = "audit_split.md"    # audit of data/{train,val,test}, printed by train.py
AUDIT_SOURCE_REPORT = "audit_aider.md"   # audit of the source dataset, before splitting
AUDIT_SMALL_DATASET = 1000               # fewer images -> report mean +/- std over seeds
AUDIT_SIZE_LEAK_WARN = 65.0              # predicting the class from original width x height
AUDIT_SIZE_LEAK_FAIL = 90.0              #   (balanced accuracy, %)
AUDIT_SIZE_BLOCK_MIN = 5                 # a "size block": >= 5 images sharing one exact size ...
AUDIT_SIZE_BLOCK_PURITY = 0.95           # ... >= 95 % of them one class ...
AUDIT_SIZE_BLOCK_MARGIN = 0.10           # ... and >= 10 points above that class's overall share
AUDIT_SIZE_BLOCK_FAIL_PCT = 35.0         # FAIL if more than 35 % of all images sit in such blocks
AUDIT_SMALL_IMAGE_WARN = 0.10            # WARN if > 10 % of images are smaller than IMG_SIZE
AUDIT_LEAK_MAX_IMAGES = 2000             # cap on images used by the source-leak test
AUDIT_LEAK_FOLDS = 5
AUDIT_LEAK_TREES = 200

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
GRADCAM_ALPHA = 0.4              # heatmap opacity in the overlay
ATTENTION_CUT = 0.5              # a pixel "counts" as attended if heat > 0.5
ATTENTION_WARN_PCT = 50.0        # above this the model is looking at the whole scene

# ---------------------------------------------------------------------------
# Output file names (joined to MODEL_DIR / REPORT_DIR, which argparse can override)
# ---------------------------------------------------------------------------
BEST_MODEL_FILE = "best_model.keras"
VARIANT_MODEL_FILES = {"frozen": "model_frozen.keras", "finetuned": "model_finetuned.keras"}
THRESHOLD_FILE = "threshold.json"
HISTORY_FILE = "history.json"
TRAINING_CURVES_FILE = "training_curves.png"
SAMPLE_BATCH_FILE = "sample_batch.png"
SPLIT_MANIFEST_FILE = "split_manifest.csv"         # data/: where every source file went
SUBSAMPLE_MANIFEST_FILE = "subsample_manifest.csv" # reports/: the no_flood images kept
SEED_SUMMARY_FILE = "seed_summary.json"

# ---------------------------------------------------------------------------
# Report figure style (light background; the train/val pair is a colour-blind
# safe categorical pair, checked with a palette validator)
# ---------------------------------------------------------------------------
PLOT_SERIES_1 = "#2a78d6"        # blue   - train curve / flood class
PLOT_SERIES_2 = "#eb6834"        # orange - validation curve / no_flood class
PLOT_SURFACE = "#fcfcfb"
PLOT_INK = "#0b0b0b"
PLOT_INK_2 = "#52514e"
PLOT_MUTED = "#898781"
PLOT_GRID = "#e1e0d9"
PLOT_AXIS = "#c3c2b7"
PLOT_DPI = 150

# ---------------------------------------------------------------------------
# GUI theme
# ---------------------------------------------------------------------------
BG = "#0f1420"; PANEL = "#161d2e"; CARD = "#1b2437"
ACCENT = "#ff7a1a"; BLUE = "#2d7ff9"; DANGER = "#ff3b30"
TEXT = "#e8edf7"; MUTED = "#8a94a8"; OK = "#28c76f"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seeds(seed=SEED):
    """Seed every random number generator we use, so runs are repeatable.

    Keras draws the default seeds of its random layers (augmentation, dropout)
    from Python's `random`, so seeding it here makes those layers repeatable too.
    GPU kernels can still add tiny run-to-run differences in the last decimals.
    """
    random.seed(seed)            # Python's built-in RNG
    np.random.seed(seed)         # NumPy (data splitting)
    tf.random.set_seed(seed)     # TensorFlow (weight init, dropout, tf.data shuffle)
    print(f"[config] Seed = {seed}  (python random, numpy, tensorflow)")


def ensure_dirs(*dirs):
    """Create output folders if missing. With no arguments: models/ and reports/."""
    for d in dirs or (MODEL_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)


def rel(path):
    """A path relative to the project root for tidy console output (absolute if outside it)."""
    path = os.path.abspath(path)
    try:
        inside = os.path.commonpath([path, PROJECT_ROOT]) == PROJECT_ROOT
    except ValueError:           # e.g. a different drive on Windows
        inside = False
    return os.path.relpath(path, PROJECT_ROOT) if inside else path


def use_report_style():
    """Apply one quiet, consistent matplotlib style to every saved report figure.

    Returns pyplot. The 'Agg' backend renders straight to files, which is all the
    scripts need and avoids opening windows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": PLOT_SURFACE,
        "axes.facecolor": PLOT_SURFACE,
        "savefig.facecolor": PLOT_SURFACE,
        "axes.edgecolor": PLOT_AXIS,
        "axes.labelcolor": PLOT_INK_2,
        "axes.titlecolor": PLOT_INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,              # gridlines behind the data
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": PLOT_GRID,
        "grid.linestyle": "-",               # solid hairlines, never dashed
        "grid.linewidth": 0.8,
        "xtick.color": PLOT_MUTED,
        "ytick.color": PLOT_MUTED,
        "text.color": PLOT_INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
    })
    return plt


# ---------------------------------------------------------------------------
# Device report - runs once, whenever any script imports config
# ---------------------------------------------------------------------------
print(f"[config] TensorFlow {tf.__version__}")
_devices = tf.config.list_physical_devices()
print("[config] Physical devices: " + ", ".join(f"{d.device_type} ({d.name})" for d in _devices))
if not tf.config.list_physical_devices("GPU"):
    print("[config] WARNING: no GPU visible - running on the CPU (slower, results still valid). "
          "On Apple Silicon, `pip install tensorflow-metal` enables the Metal GPU.")
