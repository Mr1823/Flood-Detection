"""
audit_dataset.py - check an image dataset for hidden shortcuts BEFORE training on it.

A shortcut is anything other than flood water that separates the two classes -
for example every flood photo having the same size, or the two classes coming
from different websites. A CNN learns such a property far more easily than the
content, and the test accuracy then measures the shortcut, not flood detection.

    python src/audit_dataset.py --source --balance
        the source dataset (config.SOURCE_DIR, the two class folders in
        config.SOURCE_CLASS_DIRS) BEFORE splitting, on a class-balanced random
        sample drawn with config.SEED                    -> reports/audit_aider.md
    python src/audit_dataset.py
        the finished split data/{train,val,test}         -> reports/audit_split.md
        (train.py prints its numbers at startup)
    python src/audit_dataset.py some/folder              -> reports/audit_<folder>.md

Each report is Markdown, with a .json twin holding the same numbers for scripts.
Folder layouts: <root>/<class>/* (a pool)  or  <root>/<split>/<class>/*

  1. COUNTS         class balance; small datasets need mean +/- std over seeds
  2. SIZE LEAK      can the ORIGINAL width x height alone predict the class? The
                    model never receives these numbers, but they reveal separate
                    sources, and part of them survives the resize (stretching of
                    non-square images, blur of upscaled small ones).
  3. DUPLICATES     exact (MD5) and near (256-bit difference hash) copies,
                    across splits and across classes
  4. IMAGE QUALITY  unreadable files, formats training cannot load, tiny images
  4b. SOURCE LEAK   can three content-free statistics, measured AFTER the same
                    224x224 resize training uses (blur, contrast, colour spread),
                    predict the class? The score is a FLOOR on the shortcut: a
                    fine-tuned CNN exploits the same signal far better than three
                    hand-made numbers and a small random forest.
  5. FILENAMES      one naming style per class hints at one source per class

Accuracies are balanced (mean per-class recall): chance is 50 % at any class ratio.
Verdict: FAIL, WARN or PASS from the size, duplicate and quality checks. The
audit REPORTS - it never stops training. The source-leak score has no pass line.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import sys

import numpy as np

import config  # imported before TensorFlow so its log-level setting takes effect
import tensorflow as tf
from dataset import center_crop_resize
from split_data import IGNORED_DIRS, difference_hash, md5_of, near_duplicate_pairs

# Every image-like file is audited - including formats training cannot load, so
# they are reported here instead of silently vanishing from the dataset.
AUDIT_EXTENSIONS = config.IMAGE_EXTENSIONS + (".webp", ".tif", ".tiff")
RED, GREEN, YELLOW, BOLD = "91", "92", "93", "1"
LIST_LIMIT = 5                    # example rows printed per finding


class Log:
    """Print lines (coloured in a terminal) and keep them, grouped by section, for the report."""

    def __init__(self):
        self.sections = [("", [])]            # (heading, lines)
        self.colour = sys.stdout.isatty()

    def __call__(self, text="", colour=None):
        print(f"\033[{colour}m{text}\033[0m" if colour and self.colour else text)
        if colour == BOLD:                    # bold lines are section headings
            self.sections.append((text.strip(), []))
        else:
            self.sections[-1][1].append(text)

    def markdown(self, title, facts):
        """The whole log as Markdown: a facts table, then one heading + text block per section."""
        out = [f"# {title}", "", "| | |", "|---|---|"]
        out += [f"| {key} | {value} |" for key, value in facts.items()]
        for heading, lines in self.sections:
            body = "\n".join(lines).strip("\n")
            if heading:
                out += ["", f"## {heading.strip('= ')}"]
            if body:
                out += ["", "```", body, "```"]
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Reading the dataset
# ---------------------------------------------------------------------------
def find_images(root):
    """[(split, class, path)] for every image under root, in a fixed order.

    If root contains train/, val/ and test/ only those are read, so data/pool/
    and data/rejected/ never leak into an audit of the split.
    """
    split_mode = all(os.path.isdir(os.path.join(root, s)) for s in config.SPLITS)
    found = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in IGNORED_DIRS)
        parts = os.path.relpath(dirpath, root).split(os.sep)
        if parts == ["."]:
            if split_mode:
                dirs[:] = [d for d in dirs if d in config.SPLITS]
            continue
        split, class_name = ("all", parts[0]) if len(parts) == 1 else (parts[0], parts[1])
        found += [(split, class_name, os.path.join(dirpath, name)) for name in sorted(files)
                  if not name.startswith(".") and name.lower().endswith(AUDIT_EXTENSIONS)]
    return found


def source_images(source_dir=config.SOURCE_DIR):
    """[("all", class, path)] for the source dataset's two class folders (config.SOURCE_CLASS_DIRS)."""
    found = []
    for class_name, folder in config.SOURCE_CLASS_DIRS.items():
        class_dir = os.path.join(source_dir, folder)
        if not os.path.isdir(class_dir):
            raise SystemExit(f"[audit] ERROR: source folder not found: {config.rel(class_dir)}\n"
                             "        Unzip AIDER.zip into aider_raw/ (see README).")
        found += [("all", class_name, os.path.join(class_dir, name))
                  for name in sorted(os.listdir(class_dir))
                  if not name.startswith(".") and name.lower().endswith(AUDIT_EXTENSIONS)]
    return found


def balanced_sample(found, seed):
    """Randomly sample every class down to the size of the smallest one, with a fixed seed.

    At AIDER's 8:1 ratio the "always say no_flood" baseline is 89 %, which hides
    weak shortcuts; on a 1:1 sample chance is a plain 50 %.
    """
    by_class = collections.defaultdict(list)
    for item in found:
        by_class[item[1]].append(item)
    n = min(len(items) for items in by_class.values())
    rng = np.random.default_rng(seed)
    sample = []
    for class_name in sorted(by_class):
        items = sorted(by_class[class_name], key=lambda item: item[2])   # fixed start order
        sample += [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]
    return sample


def fingerprint(root, md5_by_path):
    """SHA-256 over every (relative path, MD5): changes if any image is added, removed or edited."""
    digest = hashlib.sha256()
    for path in sorted(md5_by_path, key=lambda p: os.path.relpath(p, root)):
        digest.update(f"{os.path.relpath(path, root)}\t{md5_by_path[path]}\n".encode())
    return digest.hexdigest()


def current_fingerprint(root):
    return fingerprint(root, {path: md5_of(path) for _, _, path in find_images(root)})


def read_image(split, class_name, path):
    """Size, colour mode, format, MD5 and difference hash of one file."""
    from PIL import Image

    record = {"split": split, "cls": class_name, "path": path, "name": os.path.basename(path),
              "md5": md5_of(path),
              "loadable": path.lower().endswith(config.IMAGE_EXTENSIONS)}  # TensorFlow can read it
    try:
        with Image.open(path) as image:
            image.verify()                        # structural check: truncation, bad chunks
        with Image.open(path) as image:           # verify() leaves the file unusable, so reopen
            record.update(w=image.width, h=image.height, mode=image.mode, fmt=image.format)
        record["hash"] = difference_hash(path)
    except Exception as exc:                      # PIL raises many types for broken files
        record.update(w=0, h=0, mode="CORRUPT", fmt="CORRUPT", hash=None,
                      error=f"{type(exc).__name__}: {str(exc)[:60]}")
    return record


def balanced_accuracy(truth, guess, classes):
    """Mean of the per-class recalls, in percent. Chance is 50 % for two classes."""
    recalls = [np.mean([g == c for t, g in zip(truth, guess) if t == c]) for c in classes]
    return 100 * float(np.mean(recalls))


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def check_counts(records, classes, splits, log, issue):
    log("1. COUNTS", BOLD)
    counts = {s: {c: sum(r["split"] == s and r["cls"] == c for r in records) for c in classes}
              for s in splits}
    for s in splits:
        log(f"   {s}: " + "  ".join(f"{c}={n}" for c, n in counts[s].items()))
    totals = [sum(r["cls"] == c for r in records) for c in classes]
    ratio = max(totals) / max(min(totals), 1)
    log(f"   class ratio {ratio:.1f}:1")
    if ratio > 3:
        issue("WARN", "counts", f"Classes are imbalanced {ratio:.1f}:1 - train with class weights and "
              "judge by AUC / balanced metrics, not plain accuracy.")
    if len(records) < config.AUDIT_SMALL_DATASET:
        issue("WARN", "counts", f"Only {len(records)} images - expect unstable results; report "
              "mean +/- std over several seeds, not a single number.")
    return counts


def check_size_leak(records, classes, log, issue):
    log("\n2. SIZE LEAK (original dimensions)", BOLD)
    sized = [r for r in records if r["w"]]
    groups = collections.defaultdict(collections.Counter)
    for r in sized:
        groups[(r["w"], r["h"])][r["cls"]] += 1
    overall = collections.Counter(r["cls"] for r in sized)
    majority = overall.most_common(1)[0][0]

    # Leave-one-out: guess each image's class from the OTHER images of exactly its size.
    guesses = []
    for r in sized:
        others = groups[(r["w"], r["h"])].copy()
        others[r["cls"]] -= 1                     # hold this image out
        others = {k: v for k, v in others.items() if v > 0}
        guesses.append(max(others, key=others.get) if others else majority)
    truth = [r["cls"] for r in sized]
    balanced = balanced_accuracy(truth, guesses, classes)
    plain = 100 * np.mean([t == g for t, g in zip(truth, guesses)])
    log(f"   balanced accuracy from width x height alone : {balanced:5.1f}%   (chance 50.0%)")
    log(f"   plain accuracy                              : {plain:5.1f}%   "
        f"(always '{majority}': {100 * overall[majority] / len(sized):.1f}%)")
    for c in classes:
        log(f"      {c:<22} {len({(r['w'], r['h']) for r in sized if r['cls'] == c}):>5} distinct sizes")

    # Large blocks of images sharing one exact size and (almost) one label.
    blocks = []
    for size, count in groups.items():
        total = sum(count.values())
        top, n = count.most_common(1)[0]
        share = n / total
        if (total >= config.AUDIT_SIZE_BLOCK_MIN and share >= config.AUDIT_SIZE_BLOCK_PURITY
                and share - overall[top] / len(sized) >= config.AUDIT_SIZE_BLOCK_MARGIN):
            blocks.append((total, size, top, share))
    block_pct = 100 * sum(b[0] for b in blocks) / len(sized)
    log(f"   images in a large single-class size block   : {block_pct:5.1f}%")
    for total, size, top, share in sorted(blocks, reverse=True)[:3]:
        log(f"      {total:>5} images at {size[0]}x{size[1]} -> {100 * share:.0f}% '{top}'")

    if balanced >= config.AUDIT_SIZE_LEAK_FAIL or block_pct > config.AUDIT_SIZE_BLOCK_FAIL_PCT:
        log(f"   FAIL: image size gives the label away ({balanced:.0f}% balanced from size alone, "
            f"{block_pct:.0f}% of images in single-class size blocks).", RED)
        issue("FAIL", "size_leak", f"{block_pct:.0f}% of images sit in single-class size blocks; "
              f"dimensions alone score {balanced:.0f}% balanced accuracy.")
    elif balanced >= config.AUDIT_SIZE_LEAK_WARN:
        log(f"   WARNING: size carries real signal ({balanced:.0f}% balanced).", YELLOW)
        issue("WARN", "size_leak", f"Dimensions alone score {balanced:.0f}% balanced accuracy.")
    else:
        log("   OK: image dimensions do not give the class away.", GREEN)
    return {"balanced_accuracy": balanced, "plain_accuracy": plain, "block_pct": block_pct,
            "blocks": [{"images": t, "size": list(s), "class": c, "share": sh}
                       for t, s, c, sh in sorted(blocks, reverse=True)]}


def check_duplicates(records, pool_mode, log, issue):
    log("\n3. DUPLICATES", BOLD)
    by_md5 = collections.defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)
    exact = [g for g in by_md5.values() if len(g) > 1]
    exact_split = [g for g in exact if len({r["split"] for r in g}) > 1]
    exact_class = [g for g in exact if len({r["cls"] for r in g}) > 1]

    hashed = [r for r in records if r.get("hash") is not None]
    near = [(hashed[i], hashed[j], bits) for i, j, bits in
            near_duplicate_pairs([r["hash"] for r in hashed], config.NEAR_DUP_MAX_BITS)
            if hashed[i]["md5"] != hashed[j]["md5"]]          # exact copies are counted above
    near_split = [p for p in near if p[0]["split"] != p[1]["split"]]
    near_class = [p for p in near if p[0]["cls"] != p[1]["cls"]]

    def show(a, b, bits=None):
        extra = f"  ({bits}/256 bits differ)" if bits is not None else ""
        log(f"      {a['split']}/{a['cls']}/{a['name'][:40]}  ~=  {b['split']}/{b['cls']}/{b['name'][:40]}{extra}")

    log(f"   exact copies (MD5)                     : {sum(len(g) - 1 for g in exact)}")
    log(f"   near copies (256-bit hash, <= {config.NEAR_DUP_MAX_BITS} bits) : {len(near)}")
    for a, b, bits in near[:LIST_LIMIT]:
        show(a, b, bits)
    if not pool_mode:
        bad = len(exact_split) + len(near_split)
        log(f"   copies across splits                   : {bad}  "
            + ("<- test set is contaminated" if bad else "ok"), RED if bad else GREEN)
        if bad:
            issue("FAIL", "duplicates", f"{bad} photo(s) appear in more than one split - the test "
                  "score would be inflated. Re-run src/split_data.py.")
    conflicts = len(exact_class) + len(near_class)
    log(f"   copies across classes                  : {conflicts}  "
        + ("<- contradictory labels" if conflicts else "ok"), RED if conflicts else GREEN)
    for a, b, bits in near_class[:LIST_LIMIT]:
        show(a, b, bits)
    if conflicts:
        issue("WARN" if pool_mode else "FAIL", "duplicates",
              f"{conflicts} photo(s) appear under both classes"
              + (" - split_data.py drops them." if pool_mode else " - contradictory labels."))
    if pool_mode and (exact or near):
        log("   (split_data.py keeps one copy of each before splitting)")
    return {"exact_extra_copies": sum(len(g) - 1 for g in exact), "near_pairs": len(near),
            "across_splits": len(exact_split) + len(near_split), "across_classes": conflicts}


def check_quality(records, classes, pool_mode, log, issue):
    log("\n4. IMAGE QUALITY", BOLD)
    corrupt = [r for r in records if r["mode"] == "CORRUPT"]
    unloadable = [r for r in records if not r["loadable"] and r["mode"] != "CORRUPT"]
    min_side = min(config.IMG_SIZE)
    tiny = [r for r in records if r["w"] and min(r["w"], r["h"]) < min_side]
    log(f"   unreadable files                 : {len(corrupt)}")
    for r in corrupt[:LIST_LIMIT]:
        log(f"      {config.rel(r['path'])}  {r['error']}")
    log(f"   formats TensorFlow cannot load   : {len(unloadable)}")
    for r in unloadable[:LIST_LIMIT]:
        log(f"      {config.rel(r['path'])}")
    tiny_by_class = {c: sum(r["cls"] == c for r in tiny) for c in classes}
    log(f"   smaller than {min_side}px              : {len(tiny)} "
        f"({100 * len(tiny) / len(records):.0f}%)  by class: {tiny_by_class}")
    log(f"   formats: {dict(collections.Counter(r['fmt'] for r in records))}")
    modes = collections.Counter(r["mode"] for r in records)
    log(f"   colour modes: {dict(modes)}" + ("  -> dataset.py decodes all as 3-channel RGB"
                                             if len(modes) > 1 else ""))
    if corrupt or unloadable:
        n = len(corrupt) + len(unloadable)
        issue("WARN" if pool_mode else "FAIL", "quality",
              f"{n} file(s) training cannot read"
              + (" - split_data.py skips them." if pool_mode else " - delete or re-split."))
    if len(tiny) > config.AUDIT_SMALL_IMAGE_WARN * len(records):
        issue("WARN", "quality", f"{len(tiny)} images are smaller than {min_side}px and will be "
              f"blurry when upscaled (by class: {tiny_by_class}).")
    return {"unreadable": len(corrupt), "unloadable": len(unloadable),
            "smaller_than_input": len(tiny), "smaller_by_class": tiny_by_class,
            "colour_modes": dict(modes)}


FRAMINGS = {
    "squash": "whole image squashed to 224x224 (as in the earlier audits)",
    "centre_crop": "centred 3:4-4:3 crop, then 224x224 (what the model receives)",
}


def content_free_features(path, framing):
    """Three statistics that say nothing about WHAT is in the picture: blur (variance
    of the Laplacian), contrast, and colour spread. Measured after TensorFlow's
    decode and one of two framings - see FRAMINGS."""
    image = tf.io.decode_image(tf.io.read_file(path), channels=config.CHANNELS,
                               expand_animations=False)
    if framing == "squash":
        rgb = tf.image.resize(image, config.IMG_SIZE, method="bilinear").numpy()
    else:
        rgb = center_crop_resize(image).numpy()
    grey = rgb @ np.array([0.299, 0.587, 0.114])           # luminance
    laplacian = (grey[1:-1, :-2] + grey[1:-1, 2:] + grey[:-2, 1:-1] + grey[2:, 1:-1]
                 - 4 * grey[1:-1, 1:-1])
    return [laplacian.var(), rgb.std(), rgb.reshape(-1, 3).std(axis=0).mean()]


def check_source_leak(records, classes, seed, log, issue):
    log("\n4b. SOURCE LEAK (content-free statistics after framing to 224x224)", BOLD)
    if len(classes) != 2:
        log("   skipped - binary datasets only")
        return None
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    usable = [r for r in records if r["loadable"] and r["mode"] != "CORRUPT"]
    rng = np.random.default_rng(seed)
    if len(usable) > config.AUDIT_LEAK_MAX_IMAGES:            # stratified random sample
        keep = []
        for c in classes:
            members = [r for r in usable if r["cls"] == c]
            n = round(config.AUDIT_LEAK_MAX_IMAGES * len(members) / len(usable))
            keep += [members[i] for i in sorted(rng.choice(len(members), n, replace=False))]
        usable = keep
    folds = StratifiedKFold(config.AUDIT_LEAK_FOLDS, shuffle=True, random_state=seed)
    names = ("blur", "contrast", "colour_spread")
    log(f"   A random forest sees only blur, contrast and colour spread (chance = 50.0%,")
    log("   balanced accuracy, 5-fold cross-validation):")
    result = {"note": "lower bound on the shortcut, not a measurement of it; no pass line"}
    for framing, description in FRAMINGS.items():
        features, labels = [], []
        for r in usable:
            try:
                features.append(content_free_features(r["path"], framing))
                labels.append(classes.index(r["cls"]))
            except (tf.errors.OpError, ValueError):
                continue
        X, y = np.array(features), np.array(labels)

        def score(columns):
            forest = RandomForestClassifier(n_estimators=config.AUDIT_LEAK_TREES,
                                            class_weight="balanced", random_state=seed)
            return 100 * cross_val_score(forest, X[:, columns], y, cv=folds,
                                         scoring="balanced_accuracy").mean()

        combined = score([0, 1, 2])
        single = {name: score([k]) for k, name in enumerate(names)}
        result[framing] = {"images": int(len(y)), "balanced_accuracy": combined,
                           "single_feature": single}
        log(f"      {description}")
        log(f"         {len(y)} images: {combined:5.1f}%   ("
            + ", ".join(f"{k} {v:.1f}%" for k, v in single.items()) + ")")
    result["balanced_accuracy"] = result["squash"]["balanced_accuracy"]
    log("   No pass line: the score is reported, not judged. It is a FLOOR on the")
    log("   shortcut, not a measurement of it - a fine-tuned CNN exploits the same")
    log("   signal far more effectively than three numbers and a random forest.")
    return result


def check_filenames(records, classes, log):
    log("\n5. FILENAME PATTERNS", BOLD)
    result = {}
    for c in classes:
        names = [r["name"] for r in records if r["cls"] == c]
        prefix, n = collections.Counter(name[:6].lower() for name in names).most_common(1)[0]
        result[c] = {"example": names[0], "top_prefix": prefix, "share": n / len(names)}
        log(f"   {c:<22} e.g. {names[0][:38]:<40} prefix '{prefix}' on {100 * n / len(names):.0f}%")
    log("   -> one naming style per class suggests the classes came from different sources.")
    return result


# ---------------------------------------------------------------------------
# Running the audit, and the gate train.py uses
# ---------------------------------------------------------------------------
def _require_split(data_dir):
    """data/ must hold train/val/test before it can be audited or trained on."""
    if not all(os.path.isdir(os.path.join(data_dir, s)) for s in config.SPLITS):
        raise SystemExit(f"[audit] ERROR: {config.rel(data_dir)} has no train/val/test split yet.\n"
                         "        Audit the pool first:  python src/audit_dataset.py data/pool\n"
                         "        then split it:         python src/split_data.py")


def default_report_path(root):
    if os.path.abspath(root) == os.path.abspath(config.DATA_DIR):
        return os.path.join(config.REPORT_DIR, config.AUDIT_SPLIT_REPORT)
    return os.path.join(config.REPORT_DIR, f"audit_{os.path.basename(os.path.abspath(root))}.md")


def json_twin(report_path):
    """reports/audit_x.md -> reports/audit_x.json (same numbers, for scripts)."""
    return os.path.splitext(report_path)[0] + ".json"


def audit(found, root, report_path, seed=config.SEED, sampling="all images"):
    """Run every check on `found` [(split, class, path)] and save the Markdown + JSON report."""
    log = Log()
    issues = []

    def issue(severity, check, message):
        issues.append({"severity": severity, "check": check, "message": message})

    if not found:
        raise SystemExit(f"[audit] ERROR: no images found under {config.rel(root)}")
    print(f"[audit] Reading {len(found)} files ...")
    records = [read_image(*item) for item in found]
    classes = sorted({r["cls"] for r in records})
    splits = sorted({r["split"] for r in records})
    pool_mode = splits == ["all"]

    log(f"=== DATASET AUDIT: {config.rel(root)} ({'pool' if pool_mode else 'split'}) ===", BOLD)
    log(f"{len(records)} images ({sampling}) | {len(classes)} classes: {', '.join(classes)} | "
        f"splits: {', '.join(splits)}\n")
    report = {
        "target": config.rel(root), "mode": "pool" if pool_mode else "split",
        "sampling": sampling, "seed": seed,
        "audited_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint(root, {r["path"]: r["md5"] for r in records}),
        "n_images": len(records), "classes": classes,
        "counts": check_counts(records, classes, splits, log, issue),
        "size_leak": check_size_leak(records, classes, log, issue),
        "duplicates": check_duplicates(records, pool_mode, log, issue),
        "quality": check_quality(records, classes, pool_mode, log, issue),
        "source_leak": check_source_leak(records, classes, seed, log, issue),
        "filenames": check_filenames(records, classes, log),
    }
    verdict = ("FAIL" if any(i["severity"] == "FAIL" for i in issues)
               else "WARN" if issues else "PASS")
    report.update(issues=issues, verdict=verdict)

    log("\n=== VERDICT ===", BOLD)
    log(f"{verdict}" + (f" - {len(issues)} issue(s):" if issues else " - nothing alarming."),
        {"FAIL": RED, "WARN": YELLOW, "PASS": GREEN}[verdict])
    for n, i in enumerate(issues, 1):
        log(f"   {n}. [{i['severity']}] {i['message']}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    facts = {"Dataset": f"`{config.rel(root)}`", "Images audited": f"{len(records)} ({sampling})",
             "Seed": seed, "Audited": report["audited_at"], "Verdict": f"**{verdict}**",
             "Fingerprint": f"`{report['fingerprint'][:16]}...`"}
    with open(report_path, "w") as handle:
        handle.write(log.markdown("Dataset audit", facts))
    with open(json_twin(report_path), "w") as handle:
        json.dump(report, handle, indent=2, default=float)
    print(f"\n[audit] Saved {config.rel(report_path)} (+ {os.path.basename(json_twin(report_path))})")
    return report


def summarise_audit(data_dir=config.DATA_DIR, report_path=None):
    """Print the split audit's numbers for train.py. Reporting only - never blocks training.

    Returns a short record for history.json, saying whether the audit is missing,
    out of date (data/ changed since), or current - and its verdict and leak score.
    """
    report_path = json_twin(report_path or os.path.join(config.REPORT_DIR, config.AUDIT_SPLIT_REPORT))
    if not os.path.isfile(report_path):
        print(f"[audit] No audit of data/ found ({config.rel(report_path)}) - continuing. "
              "To produce one:  python src/audit_dataset.py")
        return {"status": "missing"}
    with open(report_path) as handle:
        report = json.load(handle)
    current = report.get("fingerprint") == current_fingerprint(data_dir)
    if not current:
        print("[audit] NOTE: data/ has changed since the audit below was made - "
              "re-run python src/audit_dataset.py for current numbers.")
    leak = report.get("source_leak") or {}
    print(f"[audit] {report['target']} audited {report['audited_at']} ({report['n_images']} images): "
          f"verdict {report['verdict']} (reported, not enforced)")
    for framing, description in FRAMINGS.items():
        if framing in leak:
            print(f"[audit]   content-free score, {description}: "
                  f"{leak[framing]['balanced_accuracy']:.1f}% (chance 50%, a lower bound)")
    for i in report["issues"]:
        print(f"[audit]   [{i['severity']}] {i['message']}")
    return {"status": "current" if current else "out of date", "verdict": report["verdict"],
            "audited_at": report["audited_at"], "fingerprint": report["fingerprint"],
            "source_leak": {k: leak[k]["balanced_accuracy"] for k in FRAMINGS if k in leak}}


def parse_args():
    parser = argparse.ArgumentParser(description="Check an image dataset for shortcuts before training.")
    parser.add_argument("root", nargs="?", default=config.DATA_DIR,
                        help="dataset folder (default: data/, i.e. the train/val/test split)")
    parser.add_argument("--source", action="store_true",
                        help="audit the source dataset (config.SOURCE_DIR) instead of a folder")
    parser.add_argument("--balance", action="store_true",
                        help="audit a random class-balanced sample (drawn with --seed)")
    parser.add_argument("--report", default=None,
                        help="Markdown report path (default: reports/audit_aider.md with --source, "
                             "reports/audit_split.md for data/, else reports/audit_<folder>.md)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    config.set_seeds(args.seed)
    if args.source:
        root, found = config.SOURCE_DIR, source_images()
        report_path = args.report or os.path.join(config.REPORT_DIR, config.AUDIT_SOURCE_REPORT)
    else:
        if not os.path.isdir(args.root):
            raise SystemExit(f"[audit] ERROR: folder not found: {args.root}")
        if os.path.abspath(args.root) == os.path.abspath(config.DATA_DIR):
            _require_split(args.root)     # data/ itself means "the split", never other folders
        root, found = args.root, find_images(args.root)
        report_path = args.report or default_report_path(args.root)
    sampling = "all images"
    if args.balance:
        before = collections.Counter(item[1] for item in found)
        found = balanced_sample(found, args.seed)
        sampling = (f"class-balanced random sample, seed {args.seed}, from "
                    + ", ".join(f"{c} {n}" for c, n in sorted(before.items())))
    audit(found, root, report_path, args.seed, sampling)


if __name__ == "__main__":
    main()
