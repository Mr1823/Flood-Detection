"""
dataset.py - tf.data input pipelines for the flood classifier.

make_datasets() yields batches of
    images : float32, shape (batch, 224, 224, 3), RGB, RAW pixel values 0..255
    labels : float32, shape (batch, 1),  0.0 = no_flood,  1.0 = flood

FRAMING - the mitigation for the shortcut the dataset audit measured
    In AIDER, much of the no_flood class comes from a few capture pipelines with
    fixed frame sizes (400x360, 399x360, 240x240), while the flood photos come in
    hundreds of sizes. Squashing every image to a square keeps each pipeline's
    geometry identical in every epoch - a stable signal the model could learn
    instead of the scene. So the fixed resize is replaced:
      training        random_resized_crop(): a random window covering 70-100 % of
                      the image area, aspect ratio 3:4 to 4:3, then 224x224. No two
                      epochs see the same framing, so capture geometry stops being a
                      stable signal.
      val/test/apps   center_crop_resize(): the largest CENTRED window whose aspect
                      ratio lies in 3:4-4:3 (the whole image if it already does),
                      then 224x224. Deterministic, so evaluation is repeatable.
    A window that would not fit inside the image is shrunk, never stretched, so
    no crop ever has an aspect ratio outside 3:4-4:3 - including for AIDER's very
    wide images (up to 2.7:1), where a standard random crop falls back to
    squashing the whole picture.

WHERE THE MobileNetV2 PREPROCESSING HAPPENS - read before changing anything
    mobilenet_v2.preprocess_input (0..255 -> -1..1) is applied in exactly ONE
    place: a layer inside the model, straight after the augmentation layers
    (model.py, MobileNetV2Preprocess). It is deliberately NOT applied here:
      1. RandomBrightness and RandomContrast assume pixels in 0..255 and clip
         their output to that range. Fed -1..1 values they would clip away
         every negative pixel. So augmentation must see raw pixels, and the
         -1..1 scaling has to come after it - i.e. inside the model too.
      2. With the scaling inside the model, every consumer (evaluation,
         Grad-CAM, the app, the TFLite export) feeds plain 0..255 RGB and can
         neither forget the preprocessing nor apply it twice.
    So: no Rescaling(1./255) and no preprocess_input anywhere in this file.

Run on its own to check the pipeline before any real training:
    python src/dataset.py
It prints the class mapping, counts and batch shapes, proves the labels match
the folder names, and saves one training batch to reports/sample_batch.png.
"""
import argparse
import math
import os
from collections import Counter

import numpy as np

import config  # imported before TensorFlow so its log-level setting takes effect
import tensorflow as tf


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------
def _image_files(class_dir):
    """Image files under class_dir in a fixed order: sub-folders sorted, names sorted."""
    paths = []
    for root, _, files in sorted(os.walk(class_dir), key=lambda entry: entry[0]):
        for name in sorted(files):
            if not name.startswith(".") and name.lower().endswith(config.IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, name))
    return paths


def list_files(split, class_name, data_dir=config.DATA_DIR):
    """Ordered file paths of one class in one split, e.g. list_files("test", "flood").

    val/test are never shuffled: they yield every no_flood file in this order,
    then every flood file. So prediction i belongs to file i of
    list_files(split, "no_flood") + list_files(split, "flood").
    """
    return _image_files(os.path.join(data_dir, split, class_name))


def check_split_dirs(data_dir=config.DATA_DIR):
    """Fail loudly, with instructions, if any split/class folder is missing or empty."""
    problems = []
    for split in config.SPLITS:
        for class_name in config.CLASS_NAMES:
            folder = os.path.join(data_dir, split, class_name)
            if not os.path.isdir(folder):
                problems.append(f"missing folder: {config.rel(folder)}")
            elif not _image_files(folder):
                problems.append(f"no images in:   {config.rel(folder)}")
    if problems:
        raise FileNotFoundError(
            "The dataset is not ready:\n  " + "\n  ".join(problems)
            + "\nExpected data/{train,val,test}/{no_flood,flood}/. Unzip AIDER into aider_raw/ "
            "(see README), then run:  python src/split_data.py")


def validate_images(data_dir=config.DATA_DIR):
    """Decode every image exactly as training will, and fail loudly on any bad file.

    A corrupt or unsupported file would otherwise crash training in the middle of
    an epoch, with an error message that does not say which file was at fault.
    """
    bad = []
    for split in config.SPLITS:
        for class_name in config.CLASS_NAMES:
            for path in list_files(split, class_name, data_dir):
                try:
                    decode(tf.io.read_file(path))
                except (tf.errors.OpError, ValueError) as exc:
                    bad.append(f"{config.rel(path)}  ({type(exc).__name__})")
    if bad:
        raise ValueError(
            "TensorFlow cannot decode these images:\n  " + "\n  ".join(bad)
            + "\nDelete or replace them, or re-split with src/split_data.py (it skips unreadable files).")


# ---------------------------------------------------------------------------
# Decoding and framing (shared by training, evaluation and every app)
# ---------------------------------------------------------------------------
def decode(data):
    """Encoded file bytes -> uint8 RGB image (H, W, 3).

    channels=3 forces every file to RGB: greyscale is copied to R, G and B, an
    alpha channel is dropped and palette images are expanded.
    """
    image = tf.io.decode_image(data, channels=config.CHANNELS, expand_animations=False)
    image.set_shape([None, None, config.CHANNELS])
    return image


def _window(height, width, area_fraction, aspect):
    """(height, width) of a window covering `area_fraction` of the image with the given
    aspect ratio (width / height). If it would not fit, it is shrunk - keeping its
    aspect ratio - until it does, so a crop is never stretched."""
    h = tf.cast(height, tf.float32)
    w = tf.cast(width, tf.float32)
    crop_w = tf.sqrt(h * w * area_fraction * aspect)
    crop_h = tf.sqrt(h * w * area_fraction / aspect)
    shrink = tf.minimum(1.0, tf.minimum(w / crop_w, h / crop_h))
    crop_h = tf.clip_by_value(tf.round(crop_h * shrink), 1.0, h)
    crop_w = tf.clip_by_value(tf.round(crop_w * shrink), 1.0, w)
    return tf.cast(crop_h, tf.int32), tf.cast(crop_w, tf.int32)


def _random_offset(seed, free):
    """A random integer in 0..free (inclusive) from a stateless float draw."""
    u = tf.random.stateless_uniform([], seed)                       # 0 <= u < 1
    offset = tf.cast(tf.floor(u * tf.cast(free + 1, tf.float32)), tf.int32)
    return tf.minimum(offset, free)                                 # guard float rounding


def random_resized_crop(image, seed):
    """TRAINING framing: a random 70-100 %-area window, aspect 3:4-4:3, resized to 224x224.

    `seed` is a pair of integers; the stateless random ops make each crop depend
    only on it, so a run is repeatable even though images are cropped in parallel.
    """
    seeds = tf.random.experimental.stateless_split(seed, num=4)
    area = tf.random.stateless_uniform([], seeds[0], *config.CROP_SCALE)
    # Aspect ratio drawn on a log scale, so 3:4 and 4:3 are equally likely.
    aspect = tf.exp(tf.random.stateless_uniform([], seeds[1], math.log(config.CROP_ASPECT[0]),
                                                math.log(config.CROP_ASPECT[1])))
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    crop_h, crop_w = _window(height, width, area, aspect)
    # Random position: a float in [0, 1) scaled to the free space and floored. (The integer
    # version of stateless_uniform fails on the tensorflow-metal GPU, so it is avoided.)
    top = _random_offset(seeds[2], height - crop_h)
    left = _random_offset(seeds[3], width - crop_w)
    crop = tf.image.crop_to_bounding_box(image, top, left, crop_h, crop_w)
    return tf.image.resize(crop, config.IMG_SIZE, method="bilinear")


def center_crop_resize(image):
    """EVALUATION / INFERENCE framing: the largest centred window with an aspect ratio
    in 3:4-4:3 (the whole image when it already is), resized to 224x224."""
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    aspect = tf.clip_by_value(tf.cast(width, tf.float32) / tf.cast(height, tf.float32),
                              *config.CROP_ASPECT)
    crop_h, crop_w = _window(height, width, 1.0, aspect)
    crop = tf.image.crop_to_bounding_box(image, (height - crop_h) // 2, (width - crop_w) // 2,
                                         crop_h, crop_w)
    return tf.image.resize(crop, config.IMG_SIZE, method="bilinear")


def load_for_model(path_or_bytes):
    """One image for inference: (cropped RGB uint8 for display, float32 model input).

    Both are the centred crop the model sees, so an overlay drawn on the display
    image lines up with what the model looked at.
    """
    data = (path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray))
            else tf.io.read_file(path_or_bytes))
    model_input = center_crop_resize(decode(data))
    display = tf.cast(tf.clip_by_value(tf.round(model_input), 0, 255), tf.uint8)
    return display.numpy(), model_input.numpy()


def spread_sample(ds, n=32):
    """n images spread evenly over a dataset's files (so both classes appear), loaded
    straight from disk. Used for the GPU-vs-CPU check: taking one batch from the
    dataset itself would read its cache only partially, which TensorFlow then discards."""
    paths = ds.file_paths
    picks = np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)
    return np.stack([load_for_model(paths[i])[1] for i in picks])


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def _split_files(data_dir, split):
    """(paths, labels) for one split: all no_flood files (label 0), then all flood (1)."""
    paths, labels = [], []
    for index, class_name in enumerate(config.CLASS_NAMES):   # pins no_flood -> 0, flood -> 1
        files = list_files(split, class_name, data_dir)
        paths += files
        labels += [index] * len(files)
    labels = np.array(labels, dtype=np.float32).reshape(-1, 1)  # shape (N, 1): matches 1 sigmoid output
    return paths, labels


def _training_pipeline(paths, labels, batch_size, seed):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda path, label: (tf.io.read_file(path), label),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()                     # keep the small COMPRESSED files in RAM; every epoch
                                        # decodes and crops them afresh
    ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    # One random seed pair per image, different every epoch but repeatable for a given seed.
    crop_seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)
    ds = tf.data.Dataset.zip((ds, crop_seeds))
    ds = ds.map(lambda item, crop_seed: (random_resized_crop(decode(item[0]), crop_seed), item[1]),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _evaluation_pipeline(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda path, label: (center_crop_resize(decode(tf.io.read_file(path))), label),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()                     # deterministic, so the finished 224x224 images are cached
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)   # never shuffled: order = file order


def make_datasets(data_dir=config.DATA_DIR, img_size=config.IMG_SIZE,
                  batch_size=config.BATCH_SIZE, seed=config.SEED, validate=True):
    """Return (train_ds, val_ds, test_ds, class_names).

    Images are raw 0..255 RGB float32 at img_size; see the module docstring.
    Every dataset carries .file_paths and .class_names; for val/test the file
    order matches the order of the predictions.
    """
    if tuple(img_size) != tuple(config.IMG_SIZE):
        raise ValueError(f"img_size {img_size} differs from config.IMG_SIZE {config.IMG_SIZE}; "
                         "change it in config.py so training and inference stay identical.")
    check_split_dirs(data_dir)
    if validate:
        validate_images(data_dir)
    datasets = []
    for split in config.SPLITS:
        paths, labels = _split_files(data_dir, split)
        print(f"[data] {split}: {len(paths)} files belonging to {len(config.CLASS_NAMES)} classes")
        ds = (_training_pipeline(paths, labels, batch_size, seed) if split == "train"
              else _evaluation_pipeline(paths, labels, batch_size))
        ds.file_paths, ds.class_names = paths, list(config.CLASS_NAMES)
        datasets.append(ds)
    return (*datasets, list(config.CLASS_NAMES))


def build_augmentation():
    """Random augmentation for TRAINING batches only.

    These layers are placed inside the model graph (model.py). Keras switches
    them on only when training=True, i.e. inside model.fit(); in evaluate() and
    predict() they pass images through unchanged, so inference is never augmented.
    There is no RandomZoom: the random crop in the input pipeline already varies scale.
    """
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            layers.RandomFlip(config.AUG_FLIP),            # left-right mirror only, never vertical
            layers.RandomRotation(config.AUG_ROTATION),    # up to +/-18 deg; corners filled by reflection
            layers.RandomBrightness(config.AUG_BRIGHTNESS,
                                    value_range=config.PIXEL_RANGE),  # needs raw 0..255 pixels
            layers.RandomContrast(config.AUG_CONTRAST),    # clips to 0..255, so it needs raw pixels too
        ],
        name="augmentation",
    )


def compute_class_weights(train_dir=os.path.join(config.DATA_DIR, "train")):
    """Class weights for model.fit, e.g. {0: 0.76, 1: 1.45}.

    "Balanced" weighting: weight_c = n_total / (n_classes * n_c). The rarer class
    gets a weight above 1, so each class contributes equally to the training loss.
    """
    counts = [len(_image_files(os.path.join(train_dir, name))) for name in config.CLASS_NAMES]
    if min(counts) == 0:
        raise FileNotFoundError(f"A class folder in {config.rel(train_dir)} is empty: "
                                f"{dict(zip(config.CLASS_NAMES, counts))}")
    total = sum(counts)
    return {index: total / (len(counts) * count) for index, count in enumerate(counts)}


# ---------------------------------------------------------------------------
# Checks and inspection (used by `python src/dataset.py` and by train.py)
# ---------------------------------------------------------------------------
def describe(train_ds, class_names, data_dir=config.DATA_DIR):
    """Print the class mapping, image counts and the shape of one training batch."""
    if class_names != config.CLASS_NAMES:
        raise ValueError(f"Unexpected class order {class_names}; expected {config.CLASS_NAMES}")
    mapping = ", ".join(f"{name} -> {index}" for index, name in enumerate(class_names))
    print(f"\n[data] Class mapping: {mapping}   (positive class: flood = 1)")

    print(f"[data] Images per split:  {'no_flood':>8} {'flood':>6} {'total':>6}")
    for split in config.SPLITS:
        n = [len(list_files(split, name, data_dir)) for name in class_names]
        print(f"[data]   {split:<17}{n[0]:>8} {n[1]:>6} {sum(n):>6}")

    images, labels = next(iter(train_ds))
    flat = labels.numpy().ravel()
    print(f"[data] One train batch: images {tuple(images.shape)} {images.dtype.name}, "
          f"labels {tuple(labels.shape)} {labels.dtype.name}")
    print(f"[data]   pixel range {float(tf.reduce_min(images)):.1f} .. {float(tf.reduce_max(images)):.1f} "
          "(raw 0..255 - preprocess_input runs inside the model)")
    print(f"[data]   labels: {int((flat == 0).sum())} no_flood, {int((flat == 1).sum())} flood")


def check_labels_match_folders(ds, split, data_dir=config.DATA_DIR):
    """Prove that labels come from the folders: no_flood/ -> 0 and flood/ -> 1, in file order.

    Only meaningful for the unshuffled val/test datasets.
    """
    expected_paths, expected_labels = [], []
    for index, class_name in enumerate(config.CLASS_NAMES):
        files = list_files(split, class_name, data_dir)
        expected_paths += files
        expected_labels += [index] * len(files)
    labels = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int).tolist()
    same_order = ([os.path.abspath(p) for p in ds.file_paths]
                  == [os.path.abspath(p) for p in expected_paths])
    if not same_order or labels != expected_labels:
        raise AssertionError(f"{split}: dataset order or labels do not match the class folders - "
                             "predictions would be matched to the wrong files.")
    print(f"[data] {split}: order and labels match the folders "
          f"({expected_labels.count(0)} files from no_flood/ -> 0, "
          f"then {expected_labels.count(1)} from flood/ -> 1)")


def image_mode_counts(data_dir=config.DATA_DIR):
    """Count the colour modes on disk (RGB, RGBA, L = greyscale, P = palette, ...)."""
    from PIL import Image

    counts = Counter()
    for split in config.SPLITS:
        for class_name in config.CLASS_NAMES:
            for path in list_files(split, class_name, data_dir):
                with Image.open(path) as image:      # reads the header only
                    counts[image.mode] += 1
    return counts


def save_sample_batch(train_ds, class_names, out_path, augmentation=None, columns=8):
    """Save one training batch - random crop plus augmentation, as the model sees it."""
    plt = config.use_report_style()
    from matplotlib.patches import Patch

    images, labels = next(iter(train_ds))
    if augmentation is not None:
        images = augmentation(images, training=True)   # training=True switches the random layers on
    images = np.clip(np.asarray(images), *config.PIXEL_RANGE).astype("uint8")
    labels = np.asarray(labels).astype(int).ravel()
    frame = {1: config.PLOT_SERIES_1, 0: config.PLOT_SERIES_2}   # flood blue, no_flood orange

    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.9, rows * 2.1 + 0.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, image, label in zip(axes, images, labels):
        ax.axis("on")
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():               # coloured frame = class, text = name + number
            spine.set_visible(True)
            spine.set_color(frame[label])
            spine.set_linewidth(4)
        ax.set_title(f"{class_names[label]} = {label}", fontsize=9, pad=3)

    fig.suptitle("One training batch (random crop + augmentation) - does every label match its picture?",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.legend(handles=[Patch(color=frame[1], label="flood = 1"),
                        Patch(color=frame[0], label="no_flood = 0")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect the input pipeline and save one training batch to reports/.")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="folder holding train/ val/ test/")
    parser.add_argument("--report-dir", default=config.REPORT_DIR, help="where sample_batch.png goes")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    config.set_seeds(args.seed)
    config.ensure_dirs(args.report_dir)

    train_ds, val_ds, test_ds, class_names = make_datasets(
        args.data_dir, config.IMG_SIZE, args.batch_size, args.seed)
    describe(train_ds, class_names, args.data_dir)
    check_labels_match_folders(val_ds, "val", args.data_dir)
    check_labels_match_folders(test_ds, "test", args.data_dir)

    weights = compute_class_weights(os.path.join(args.data_dir, "train"))
    print("[data] Class weights: " + ", ".join(f"{class_names[i]} ({i}) = {w:.3f}" for i, w in weights.items()))
    modes = image_mode_counts(args.data_dir)
    print("[data] Colour modes on disk: " + ", ".join(f"{m} {n}" for m, n in modes.most_common())
          + "  -> all decoded as 3-channel RGB")

    out_path = os.path.join(args.report_dir, config.SAMPLE_BATCH_FILE)
    save_sample_batch(train_ds, class_names, out_path, augmentation=build_augmentation())
    print(f"[data] Saved {config.rel(out_path)}")


if __name__ == "__main__":
    main()
