"""
gradcam.py - Grad-CAM: which part of the image drove the flood score.

compute_gradcam(model, img_array, layer_name="Conv_1")
    1. grad_model maps the input to [Conv_1 feature maps (7x7x1280), flood score]
    2. GradientTape: gradient of the flood score w.r.t. those feature maps
    3. weight of each of the 1280 channels = mean of its gradient over the 7x7 grid
    4. weighted sum of the channels, ReLU (keep only evidence FOR flood), normalise to 0..1
    5. resize to the image size; the RAW heatmap (float 0..1) is returned
    The score differentiated is the logit z just before the sigmoid (P(flood) = sigmoid(z)).
    Since d sigmoid(z) = p(1-p) dz, the map is identical up to one positive factor,
    which the 0..1 normalisation removes. But for a confident prediction p rounds to
    exactly 1.0 in float32, p(1-p) becomes 0 and the map would be empty - the logit
    avoids that underflow.

overlay(image, heatmap, alpha=0.4)   jet colour map blended over the image
attention_area(heatmap, cut=0.5)     % of pixels whose heat exceeds `cut`. Above 50 %
    the model is reacting to the whole scene rather than localising water, which
    would suggest background or source cues - so a warning is printed.

CLI:  python src/gradcam.py
    reports/gradcam_grid.png    4 correct + 4 misclassified test images, original and overlay
    reports/gradcam_stats.json  attention area over the WHOLE test set (mean, share above 50 %),
                                plus a deletion test: does removing the most-attended 25 % of
                                pixels hurt P(flood) more than removing the least-attended 25 %?
"""
import argparse
import json
import os

import numpy as np

import config  # prints the TensorFlow version and devices; imported before TensorFlow
import tensorflow as tf
from dataset import list_files, load_for_model
from model import load_trained_model

_GRAD_MODELS = {}                 # built once per (model, layer), reused for every image


def _grad_model(model, layer_name):
    key = (id(model), layer_name)
    if key not in _GRAD_MODELS:
        head = model.layers[-1]   # Dense(1, sigmoid): its input is the 128-d feature vector
        _GRAD_MODELS[key] = (tf.keras.Model(model.inputs,
                                            [model.get_layer(layer_name).output, head.input]),
                             head)
    return _GRAD_MODELS[key]


def explain(model, img_array, layer_name=config.GRADCAM_LAYER):
    """(P(flood), 7x7 Grad-CAM map in 0..1) from ONE forward pass. img_array: 224x224x3, 0..255."""
    x = np.asarray(img_array, dtype=np.float32)
    if x.ndim == 3:
        x = x[None]                                   # add the batch axis
    grad_model, head = _grad_model(model, layer_name)
    with tf.GradientTape() as tape:
        conv_out, features = grad_model([x], training=False)  # training=False: no augmentation
        logit = tf.matmul(features, head.kernel) + head.bias  # z, before the sigmoid
    grads = tape.gradient(logit, conv_out)                    # d z / d feature maps
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))           # one weight per channel
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
    peak = cam.max()
    cam = cam / peak if peak > 0 else np.zeros_like(cam)      # no positive evidence -> all zero
    return float(tf.sigmoid(logit)[0, 0]), cam


def compute_gradcam(model, img_array, layer_name=config.GRADCAM_LAYER, size=None):
    """Raw Grad-CAM heatmap, float 0..1, resized to `size` (default: the 224x224 model input)."""
    _, cam = explain(model, img_array, layer_name)
    return resize_heatmap(cam, size or config.IMG_SIZE)


def resize_heatmap(cam, size):
    heat = tf.image.resize(cam[..., None], size, method="bilinear").numpy()[..., 0]
    return np.clip(heat, 0.0, 1.0)


def overlay(image, heatmap, alpha=config.GRADCAM_ALPHA):
    """Blend a jet-coloured heatmap over an RGB uint8 image."""
    import matplotlib

    if heatmap.shape != image.shape[:2]:
        heatmap = resize_heatmap(heatmap, image.shape[:2])
    colours = matplotlib.colormaps["jet"](heatmap)[..., :3] * 255.0   # RGB, no BGR pitfall
    return np.clip((1 - alpha) * image + alpha * colours, 0, 255).astype(np.uint8)


def attention_area(heatmap, cut=config.ATTENTION_CUT, warn=True):
    """Percentage of the image whose heat exceeds `cut`; warns above ATTENTION_WARN_PCT."""
    area = 100.0 * float((heatmap > cut).mean())
    if warn and area > config.ATTENTION_WARN_PCT:
        print(f"[gradcam] WARNING: attention covers {area:.1f}% of the image (> "
              f"{config.ATTENTION_WARN_PCT:.0f}%) - the model is reacting to the whole scene rather "
              "than localising water, which suggests background or source cues.")
    return area


def load_threshold(model_dir):
    path = os.path.join(model_dir, config.THRESHOLD_FILE)
    if not os.path.isfile(path):
        print(f"[gradcam] {config.rel(path)} not found - using {config.DEFAULT_THRESHOLD}. "
              "Run python src/evaluate.py to choose the threshold.")
        return config.DEFAULT_THRESHOLD
    with open(path) as handle:
        return float(json.load(handle)["threshold"])


# ---------------------------------------------------------------------------
# CLI: test-set statistics and the evidence grid
# ---------------------------------------------------------------------------
def summarise(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), "mean_attention_pct": float(values.mean()),
            "median_attention_pct": float(np.median(values)),
            "share_above_warn_pct": float(100 * (values > config.ATTENTION_WARN_PCT).mean())}


DELETION_SHARE = 0.25            # share of the image greyed out in the deletion test
NEUTRAL_GREY = 127.5             # becomes 0 after preprocess_input - the network's "no signal"


def deletion_test(model, flood_items, threshold):
    """Is the flood evidence localised where Grad-CAM points, or spread over the whole scene?

    Attention AREA cannot answer this: in aerial flood photos the water often fills
    most of the frame. So, size-controlled: grey out the 25 % of pixels Grad-CAM
    ranks HIGHEST, and separately the 25 % it ranks LOWEST, and compare the drop in
    P(flood). If the model used the whole scene evenly, both deletions would hurt
    about equally. This tests localisation and faithfulness - not WHAT the evidence is.
    """
    originals, top, bottom = [], [], []
    for _, heat, _, prob, _, path in flood_items:
        _, model_input = load_for_model(path)
        # Rank pixels by heat and take exactly k from each end. (A quantile cut-off would
        # remove more than 25 % when many pixels share the value 0 after the ReLU.)
        order = np.argsort(heat, axis=None, kind="stable")
        k = int(DELETION_SHARE * heat.size)
        hi, lo = model_input.copy(), model_input.copy()
        hi.reshape(-1, config.CHANNELS)[order[-k:]] = NEUTRAL_GREY    # most-attended quarter
        lo.reshape(-1, config.CHANNELS)[order[:k]] = NEUTRAL_GREY     # least-attended quarter
        originals.append(prob)
        top.append(hi)
        bottom.append(lo)
    p_top = model.predict(np.stack(top), verbose=0).ravel()
    p_bottom = model.predict(np.stack(bottom), verbose=0).ravel()
    originals = np.array(originals)
    return {
        "images": int(len(originals)), "deleted_share": DELETION_SHARE, "fill": "neutral grey (127.5)",
        "mean_p_flood_original": float(originals.mean()),
        "mean_p_flood_most_attended_deleted": float(p_top.mean()),
        "mean_p_flood_least_attended_deleted": float(p_bottom.mean()),
        "still_flood_after_most_attended_deleted_pct": float(100 * (p_top >= threshold).mean()),
        "still_flood_after_least_attended_deleted_pct": float(100 * (p_bottom >= threshold).mean()),
    }


def plot_grid(chosen, out_path, threshold):
    plt = config.use_report_style()
    fig, axes = plt.subplots(4, 4, figsize=(14, 15.5))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for slot, item in enumerate(chosen):
        if item is None:
            axes[2 * slot].text(0.5, 0.5, "no image\nto show", ha="center",
                                va="center", fontsize=11, color=config.PLOT_MUTED)
            continue
        display, heat, label, prob, area, path = item
        pred = int(prob >= threshold)
        axes[2 * slot].imshow(display)
        axes[2 * slot].set_title(f"{os.path.basename(path)}\ntrue {config.CLASS_NAMES[label]}",
                                 fontsize=9)
        axes[2 * slot + 1].imshow(overlay(display, heat))
        axes[2 * slot + 1].set_title(f"pred {config.CLASS_NAMES[pred]}, P(flood) {prob:.3f}\n"
                                     f"attention {area:.1f}% of image", fontsize=9,
                                     color=config.PLOT_INK if pred == label else "#d03b3b")
    fig.text(0.01, 0.985, "Grad-CAM evidence (Conv_1) - rows 1-2: correct, rows 3-4: misclassified",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.965, f"Decision threshold {threshold:.2f}. Red = high flood evidence. "
             f"Attention = share of pixels with heat > {config.ATTENTION_CUT}.",
             fontsize=9.5, color=config.PLOT_INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM evidence grid and test-set attention statistics.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument("--report-dir", default=config.REPORT_DIR)
    parser.add_argument("--layer", default=config.GRADCAM_LAYER)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    config.set_seeds(args.seed)
    config.ensure_dirs(args.report_dir)
    model = load_trained_model(os.path.join(args.model_dir, config.BEST_MODEL_FILE), compile=False)
    threshold = load_threshold(args.model_dir)

    items = []                                 # every test image: (display, heat, label, p, area, path)
    for label, class_name in enumerate(config.CLASS_NAMES):
        for path in list_files("test", class_name, args.data_dir):
            display, model_input = load_for_model(path)
            prob, cam = explain(model, model_input, args.layer)
            heat = resize_heatmap(cam, display.shape[:2])
            items.append((display, heat, label, prob, attention_area(heat, warn=False), path))
    print(f"[gradcam] {len(items)} test images explained (layer {args.layer}, threshold {threshold:.2f})")

    correct = [it for it in items if int(it[3] >= threshold) == it[2]]
    wrong = sorted((it for it in items if int(it[3] >= threshold) != it[2]),
                   key=lambda it: -abs(it[3] - threshold))           # most confident mistakes first
    stats = {
        "layer": args.layer, "threshold": threshold, "cut": config.ATTENTION_CUT,
        "warn_above_pct": config.ATTENTION_WARN_PCT,
        "all_test_images": summarise([it[4] for it in items]),
        "true_flood": summarise([it[4] for it in items if it[2] == 1]),
        "true_no_flood": summarise([it[4] for it in items if it[2] == 0]),
        "predicted_flood": summarise([it[4] for it in items if it[3] >= threshold]),
        "correct": summarise([it[4] for it in correct]),
        "misclassified": summarise([it[4] for it in wrong]),
        "deletion_test_detected_floods": deletion_test(
            model, [it for it in items if it[2] == 1 and it[3] >= threshold], threshold),
    }
    with open(os.path.join(args.report_dir, "gradcam_stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2)
    for key in ("all_test_images", "true_flood", "true_no_flood", "predicted_flood", "misclassified"):
        s = stats[key]
        if s["n"]:
            print(f"[gradcam] {key:<16} n={s['n']:>3}  mean attention {s['mean_attention_pct']:5.1f}%  "
                  f"median {s['median_attention_pct']:5.1f}%  above {config.ATTENTION_WARN_PCT:.0f}%: "
                  f"{s['share_above_warn_pct']:5.1f}% of images")
    d = stats["deletion_test_detected_floods"]
    print(f"[gradcam] Deletion test on {d['images']} detected floods (grey out {100 * DELETION_SHARE:.0f}% "
          f"of pixels): mean P(flood) {d['mean_p_flood_original']:.3f} -> "
          f"{d['mean_p_flood_most_attended_deleted']:.3f} with the MOST-attended pixels removed, "
          f"{d['mean_p_flood_least_attended_deleted']:.3f} with the LEAST-attended removed; still "
          f"classed flood: {d['still_flood_after_most_attended_deleted_pct']:.1f}% vs "
          f"{d['still_flood_after_least_attended_deleted_pct']:.1f}%")

    # Grid: 2 correct floods + 2 correct no_floods (random, seeded) + up to 4 misclassified.
    rng = np.random.default_rng(args.seed)
    picks = []
    for label in (1, 0):
        pool = [it for it in correct if it[2] == label]
        picks += [pool[i] for i in sorted(rng.choice(len(pool), min(2, len(pool)), replace=False))]
    picks += [None] * (4 - len(picks))            # keep correct examples in rows 1-2
    picks += wrong[:4] + [None] * max(0, 4 - len(wrong))
    out = os.path.join(args.report_dir, "gradcam_grid.png")
    plot_grid(picks, out, threshold)
    for it in picks:
        if it is not None:
            attention_area(it[1])               # prints the >50 % warning for any shown image
    print(f"[gradcam] Saved {config.rel(out)} and gradcam_stats.json")


if __name__ == "__main__":
    main()
