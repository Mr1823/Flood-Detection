"""
split_data.py - build data/{train,val,test}/{flood,no_flood} from AIDER.

Source: config.SOURCE_DIR (aider_raw/AIDER), class folders from config.SOURCE_CLASS_DIRS
    flooded_areas/ (526 images)   -> flood
    normal/        (4,390 images) -> no_flood
AIDER's other three classes are ignored.

Steps, in this order
  1. collect every file in the two class folders
  2. skip files TensorFlow cannot decode, and blank (single-colour) images -
     each one is listed, none silently
  3. remove exact duplicates (MD5) and near-duplicates (256-bit difference hash)
     over the WHOLE pool. This runs BEFORE subsampling, so a copy can never be
     kept while its original is dropped, nor land in a different split from it.
     A picture found under BOTH classes has a contradictory label and is dropped.
  4. subsample no_flood to config.NO_FLOOD_SUBSAMPLE (1,000) with config.SEED;
     flood keeps every image. At AIDER's 8:1 ratio "always say no_flood" would
     score 89 %; about 2:1 keeps the classes comparable. The kept no_flood files
     are listed in reports/subsample_manifest.csv.
  5. stratified 70/15/15 split of each class, with config.SEED
  6. COPY (never move) into data/<split>/<class>/; data/split_manifest.csv
     records where every source file went, or why it was left out
  7. audit the finished split -> reports/audit_split.md (+ .json). The audit
     reports; it never stops the pipeline.

If data/{train,val,test}/{flood,no_flood} already exist it does nothing.
Delete data/train, data/val and data/test to re-split.

    python src/split_data.py
"""
import argparse
import csv
import hashlib
import os
import shutil

import numpy as np

import config  # imported before TensorFlow so its log-level setting takes effect
import tensorflow as tf

IGNORED_DIRS = ("__MACOSX",)     # macOS zip metadata, never real images


# ---------------------------------------------------------------------------
# Locating the source images
# ---------------------------------------------------------------------------
def split_exists(data_dir):
    """True if every split/class folder exists; refuse to continue if only some do."""
    present = [os.path.isdir(os.path.join(data_dir, split, name))
               for split in config.SPLITS for name in config.CLASS_NAMES]
    if all(present):
        return True
    if any(present):
        raise SystemExit("[split] ERROR: only some of data/{train,val,test}/{flood,no_flood} exist, "
                         "so an earlier split was interrupted. Delete data/train, data/val and "
                         "data/test, then run this script again.")
    return False


def files_in(folder):
    """All non-hidden files under folder, in a fixed (sorted) order."""
    paths = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in IGNORED_DIRS)
        # Hidden files: .DS_Store, and macOS "._name.jpg" resource forks that only look like images.
        paths += [os.path.join(root, n) for n in sorted(names) if not n.startswith(".")]
    return paths


def source_class_folders(source_dir):
    """{class name: source folder}, failing loudly if a folder is missing."""
    folders = {name: os.path.join(source_dir, config.SOURCE_CLASS_DIRS[name])
               for name in config.CLASS_NAMES}
    missing = [config.rel(f) for f in folders.values() if not os.path.isdir(f)]
    if missing:
        raise SystemExit(f"[split] ERROR: source folder(s) not found: {missing}\n"
                         "        Unzip AIDER.zip so that aider_raw/AIDER/flooded_areas and "
                         "aider_raw/AIDER/normal exist (see README).")
    return folders


# ---------------------------------------------------------------------------
# Filtering and de-duplication
# ---------------------------------------------------------------------------
def why_unusable(path):
    """None if TensorFlow decodes the file the way training will, otherwise the reason."""
    if not path.lower().endswith(config.IMAGE_EXTENSIONS):
        return f"unsupported file type '{os.path.splitext(path)[1] or '(none)'}'"
    try:
        image = tf.io.decode_image(tf.io.read_file(path), channels=config.CHANNELS,
                                   expand_animations=False)
    except (tf.errors.OpError, ValueError) as exc:
        return f"cannot be decoded ({type(exc).__name__})"
    if min(image.shape[:2]) < 1:
        return "empty image"
    if float(tf.math.reduce_std(tf.cast(image, tf.float32))) < config.BLANK_PIXEL_STD:
        return "blank image (one flat colour, no scene)"
    return None


def md5_of(path):
    """Fingerprint of the file's bytes. MD5 is fine here: we need identity, not security."""
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def deduplicate(candidates):
    """Keep one copy of every exact duplicate.

    candidates: [(class_name, path), ...] in a fixed order; the first copy wins.
    Returns ({class_name: [(path, md5), ...]}, [removed rows for the manifest]).
    """
    by_hash = {}
    for class_name, path in candidates:
        by_hash.setdefault(md5_of(path), []).append((class_name, path))

    kept = {name: [] for name in config.CLASS_NAMES}
    removed = []
    for digest, copies in by_hash.items():
        if len({class_name for class_name, _ in copies}) > 1:       # same bytes, different labels
            removed += [(class_name, path, digest, "same image under both classes - label unclear")
                        for class_name, path in copies]
            continue
        class_name, keep_path = copies[0]
        kept[class_name].append((keep_path, digest))
        removed += [(class_name, path, digest, f"exact duplicate of {config.rel(keep_path)}")
                    for _, path in copies[1:]]
    return kept, removed


def difference_hash(path, size=config.NEAR_DUP_HASH_SIZE):
    """Perceptual fingerprint: size x size bits that survive resizing and re-compression.

    The image is shrunk to (size+1) x size greyscale pixels and each bit records
    whether a pixel is brighter than its left-hand neighbour. Resizing or
    re-saving a photo barely changes these bits; a different photo flips about half.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image.draft("L", (size * 8, size * 8))      # JPEGs: decode at reduced size (much faster)
        image = ImageOps.exif_transpose(image)
        if image.mode == "P":                       # palette PNGs: avoid a transparency warning
            image = image.convert("RGBA")
        grey = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(grey, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).ravel()


_BITS_SET = np.array([bin(n).count("1") for n in range(256)], dtype=np.uint8)  # popcount table


def near_duplicate_pairs(hashes, max_bits, chunk=512):
    """All pairs (i, j, bits) with i < j whose hashes differ in <= max_bits bits.

    The number of differing bits is the Hamming distance. Comparing every pair at
    once would need N x N x 256 values (6 GB for 5,000 images), so the hashes are
    packed into bytes and compared one block of rows at a time.
    """
    packed = np.packbits(np.asarray(hashes, dtype=bool), axis=1)   # 256 bits -> 32 bytes
    pairs = []
    for start in range(0, len(packed), chunk):
        block = packed[start:start + chunk]
        bits = _BITS_SET[block[:, None, :] ^ packed[None, :, :]].sum(axis=-1, dtype=np.int32)
        rows, cols = np.nonzero(bits <= max_bits)
        for r, c in zip(rows, cols):
            if c > r + start:                       # each pair once, never an image with itself
                pairs.append((int(r + start), int(c), int(bits[r, c])))
    return pairs


def remove_near_duplicates(kept, max_bits):
    """Group images whose hashes differ in <= max_bits and keep one per group.

    The highest-resolution copy is kept. A group spanning both classes has a
    contradictory label, so all of it is dropped. Returns (kept, removed rows).
    """
    from PIL import Image

    items = [(name, path, digest) for name in config.CLASS_NAMES for path, digest in kept[name]]
    hashes = np.array([difference_hash(path) for _, path, _ in items])

    # Union-find: link every close pair, so chains of near-copies form one group.
    parent = list(range(len(items)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in near_duplicate_pairs(hashes, max_bits):
        parent[root(i)] = root(j)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(root(i), []).append(i)

    def pixel_count(path):
        with Image.open(path) as image:
            return image.width * image.height

    new_kept = {name: [] for name in config.CLASS_NAMES}
    removed = []
    for members in groups.values():
        if len({items[i][0] for i in members}) > 1:
            removed += [(items[i][0], items[i][1], items[i][2],
                         "near-duplicate found under both classes - label unclear") for i in members]
            continue
        best = max(members, key=lambda i: (pixel_count(items[i][1]), -i))   # largest; ties: first
        name, path, digest = items[best]
        new_kept[name].append((path, digest))
        removed += [(items[i][0], items[i][1], items[i][2],
                     f"near-duplicate of {config.rel(path)} "
                     f"({int((hashes[i] != hashes[best]).sum())}/{hashes.shape[1]} bits differ)")
                    for i in members if i != best]
    return new_kept, removed


# ---------------------------------------------------------------------------
# Subsampling, splitting and copying
# ---------------------------------------------------------------------------
def subsample_no_flood(kept, limit, rng):
    """Keep at most `limit` no_flood images, chosen at random; flood keeps everything.

    Returns (kept, dropped) where dropped lists the no_flood images left out.
    """
    items = sorted(kept["no_flood"])              # fixed start order: the seed alone decides
    if len(items) <= limit:
        return kept, []
    chosen = set(rng.choice(len(items), limit, replace=False).tolist())
    keep = [item for i, item in enumerate(items) if i in chosen]
    dropped = [item for i, item in enumerate(items) if i not in chosen]
    return {**kept, "no_flood": keep}, dropped


def stratified_split(kept, fractions, rng):
    """Shuffle each class with the seeded generator and cut it 70/15/15.

    Splitting each class separately (stratification) gives every split the same
    flood / no_flood balance as the whole dataset.
    """
    assignment = {split: {} for split in config.SPLITS}
    for class_name in config.CLASS_NAMES:
        items = sorted(kept[class_name])          # fixed start order: the seed alone decides
        order = rng.permutation(len(items))
        n_train = int(round(len(items) * fractions[0]))
        n_val = int(round(len(items) * fractions[1]))
        cuts = {"train": order[:n_train],
                "val": order[n_train:n_train + n_val],
                "test": order[n_train + n_val:]}
        for split, indices in cuts.items():
            if len(indices) == 0:
                raise SystemExit(f"[split] ERROR: too few '{class_name}' images ({len(items)}) "
                                 f"to give the {split} split at least one.")
            assignment[split][class_name] = [items[i] for i in indices]
    return assignment


def copy_split(assignment, data_dir):
    """Copy every file to data/<split>/<class>/ and return manifest rows."""
    rows = []
    for split in config.SPLITS:
        for class_name in config.CLASS_NAMES:
            dest_dir = os.path.join(data_dir, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for src, digest in sorted(assignment[split][class_name]):
                name = os.path.basename(src)
                dest = os.path.join(dest_dir, name)
                if os.path.exists(dest):        # two different photos with the same file name
                    stem, ext = os.path.splitext(name)
                    dest = os.path.join(dest_dir, f"{stem}_{digest[:8]}{ext}")
                shutil.copy2(src, dest)         # copy, not move: the original stays intact
                rows.append({"split": split, "class": class_name, "file": config.rel(dest),
                             "source": config.rel(src), "md5": digest, "note": ""})
    return rows


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="De-duplicate AIDER, subsample no_flood and make a stratified train/val/test split.")
    parser.add_argument("--source-dir", default=config.SOURCE_DIR,
                        help="folder holding AIDER's class folders (default: aider_raw/AIDER)")
    parser.add_argument("--data-dir", default=config.DATA_DIR,
                        help="where data/{train,val,test}/ are created (default: data/)")
    parser.add_argument("--report-dir", default=config.REPORT_DIR)
    parser.add_argument("--no-flood-limit", type=int, default=config.NO_FLOOD_SUBSAMPLE,
                        help="no_flood images kept after de-duplication")
    parser.add_argument("--fractions", type=float, nargs=3, default=config.SPLIT_FRACTIONS,
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--near-dup-bits", type=int, default=config.NEAR_DUP_MAX_BITS,
                        help="max differing hash bits to count as the same photo "
                             "(negative = skip near-duplicate removal)")
    parser.add_argument("--no-audit", action="store_true", help="skip the audit of the finished split")
    args = parser.parse_args()
    if abs(sum(args.fractions) - 1.0) > 1e-6:
        parser.error(f"--fractions must add up to 1, got {sum(args.fractions):.3f}")
    return args


def main():
    args = parse_args()
    config.set_seeds(args.seed)
    rng = np.random.default_rng(args.seed)       # one seeded generator: subsample, then split
    data_dir = os.path.abspath(args.data_dir)

    if split_exists(data_dir):
        print(f"[split] {config.rel(data_dir)}/train, val and test already exist - nothing to do.")
        return

    folders = source_class_folders(args.source_dir)
    for class_name, folder in folders.items():
        print(f"[split]   {class_name:<8} <- {config.rel(folder)}")

    # 1-2. Collect files and drop the ones TensorFlow cannot read.
    candidates, skipped = [], []
    for class_name in config.CLASS_NAMES:
        for path in files_in(folders[class_name]):
            reason = why_unusable(path)
            if reason:
                skipped.append((class_name, path, "", reason))
            else:
                candidates.append((class_name, path))
    counts = {c: sum(k == c for k, _ in candidates) for c in config.CLASS_NAMES}
    print(f"[split] Readable images: {len(candidates)} {counts}")
    for _, path, _, reason in skipped:
        print(f"[split]   skipped {config.rel(path)}  - {reason}")

    # 3. Duplicates and near-duplicates over the WHOLE pool, before any subsampling.
    kept, removed = deduplicate(candidates)
    print(f"[split] Removed {len(removed)} exact duplicate(s) (MD5 of file bytes)")
    if args.near_dup_bits >= 0:
        kept, near = remove_near_duplicates(kept, args.near_dup_bits)
        by_class = {c: sum(r[0] == c for r in near) for c in config.CLASS_NAMES}
        print(f"[split] Removed {len(near)} near-duplicate(s) (difference hash, "
              f"<= {args.near_dup_bits} of 256 bits differ): {by_class}")
        removed += near
    print("[split] After de-duplication: " + str({c: len(kept[c]) for c in config.CLASS_NAMES}))

    # 4. Subsample no_flood, with the seeded generator.
    kept, not_sampled = subsample_no_flood(kept, args.no_flood_limit, rng)
    subsample_path = os.path.join(args.report_dir, config.SUBSAMPLE_MANIFEST_FILE)
    write_csv(subsample_path,
              [{"class": "no_flood", "file": config.rel(p), "md5": d}
               for p, d in sorted(kept["no_flood"])],
              ["class", "file", "md5"])
    print(f"[split] Subsampled no_flood to {len(kept['no_flood'])} (seed {args.seed}); "
          f"kept files listed in {config.rel(subsample_path)}")

    # 5-6. Split, copy, record.
    assignment = stratified_split(kept, args.fractions, rng)
    rows = copy_split(assignment, data_dir)
    rows += [{"split": "removed", "class": c, "file": "", "source": config.rel(p), "md5": d, "note": r}
             for c, p, d, r in removed]
    rows += [{"split": "skipped", "class": c, "file": "", "source": config.rel(p), "md5": d, "note": r}
             for c, p, d, r in skipped]
    rows += [{"split": "not_sampled", "class": "no_flood", "file": "", "source": config.rel(p),
              "md5": d, "note": f"left out by the {args.no_flood_limit}-image subsample"}
             for p, d in not_sampled]
    manifest = os.path.join(data_dir, config.SPLIT_MANIFEST_FILE)
    write_csv(manifest, rows, ["split", "class", "file", "source", "md5", "note"])

    pct = "/".join(f"{round(f * 100)}" for f in args.fractions)
    print(f"[split] Stratified {pct} split, seed {args.seed}:")
    print(f"[split]   {'':<6}{'no_flood':>9}{'flood':>7}{'total':>7}")
    totals = [0, 0]
    for split in config.SPLITS:
        n = [len(assignment[split][name]) for name in config.CLASS_NAMES]
        totals = [t + x for t, x in zip(totals, n)]
        print(f"[split]   {split:<6}{n[0]:>9}{n[1]:>7}{sum(n):>7}")
    print(f"[split]   {'total':<6}{totals[0]:>9}{totals[1]:>7}{sum(totals):>7}")
    print(f"[split] Copied {sum(totals)} files into {config.rel(data_dir)}/train, val, test "
          f"(originals untouched). Manifest: {config.rel(manifest)}")

    # 7. Audit the finished split - reported, never blocking.
    if not args.no_audit:
        from audit_dataset import audit, find_images   # imported here: audit_dataset imports this file
        print("\n[split] Auditing the finished split ...")
        audit(find_images(data_dir), data_dir,
              os.path.join(args.report_dir, config.AUDIT_SPLIT_REPORT), args.seed)


if __name__ == "__main__":
    main()
