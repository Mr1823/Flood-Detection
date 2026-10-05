# Dataset audit

| | |
|---|---|
| Dataset | `data/rejected` |
| Images audited | 475 (all images) |
| Seed | 42 |
| Audited | 2026-10-04T20:46:18 |
| Verdict | **FAIL** |
| Fingerprint | `34f2cf45afd057d5...` |

## DATASET AUDIT: data/rejected (split)

```
475 images (all images) | 2 classes: flooding, normal | splits: test, train, valid
```

## 1. COUNTS

```
   test: flooding=43  normal=29
   train: flooding=168  normal=154
   valid: flooding=42  normal=39
   class ratio 1.1:1
```

## 2. SIZE LEAK (original dimensions)

```
   balanced accuracy from width x height alone :  73.6%   (chance 50.0%)
   plain accuracy                              :  75.4%   (always 'flooding': 53.3%)
      flooding                   1 distinct sizes
      normal                   136 distinct sizes
   images in a large single-class size block   :  67.4%
        253 images at 512x384 -> 100% 'flooding'
         21 images at 128x128 -> 100% 'normal'
         19 images at 100x100 -> 100% 'normal'
   FAIL: image size gives the label away (74% balanced from size alone, 67% of images in single-class size blocks).
```

## 3. DUPLICATES

```
   exact copies (MD5)                     : 1
   near copies (256-bit hash, <= 25 bits) : 5
      test/normal/stock-photo-red-dirt-road-rural-roads-in  ~=  train/normal/images (3).jpg  (16/256 bits differ)
      test/normal/village-road-surrounded-green-grass-road  ~=  train/normal/images (4).jpg  (8/256 bits differ)
      test/normal/village-roads (1).jpg  ~=  valid/normal/images (2).jpg  (16/256 bits differ)
      train/normal/drawing-line-suburban-roads (1).jpg  ~=  train/normal/drawing-line-suburban-roads.jpg  (14/256 bits differ)
      train/normal/images.jpg  ~=  valid/normal/beautiful-village-highway-stock-photo_cs  (7/256 bits differ)
   copies across splits                   : 4  <- test set is contaminated
   copies across classes                  : 0  ok
```

## 4. IMAGE QUALITY

```
   unreadable files                 : 0
   formats TensorFlow cannot load   : 1
      data/rejected/train/normal/_f11a373c-8df7-11e5-aaeb-430f0a0994d0.webp
   smaller than 224px              : 93 (20%)  by class: {'flooding': 0, 'normal': 93}
   formats: {'JPEG': 438, 'PNG': 35, 'GIF': 1, 'WEBP': 1}
   colour modes: {'RGB': 438, 'P': 13, 'RGBA': 23, 'L': 1}  -> dataset.py decodes all as 3-channel RGB
```

## 4b. SOURCE LEAK (content-free statistics after framing to 224x224)

```
   A random forest sees only blur, contrast and colour spread (chance = 50.0%,
   balanced accuracy, 5-fold cross-validation):
      whole image squashed to 224x224 (as in the earlier audits)
         474 images:  74.6%   (blur 66.7%, contrast 51.9%, colour_spread 50.1%)
      centred 3:4-4:3 crop, then 224x224 (what the model receives)
         474 images:  74.4%   (blur 64.8%, contrast 50.9%, colour_spread 51.7%)
   No pass line: the score is reported, not judged. It is a FLOOR on the
   shortcut, not a measurement of it - a fine-tuned CNN exploits the same
   signal far more effectively than three numbers and a random forest.
```

## 5. FILENAME PATTERNS

```
   flooding               e.g. image_211.jpg                            prefix 'image_' on 100%
   normal                 e.g. Sampling-area-urban-suburban-rural-loc   prefix 'ia_100' on 23%
   -> one naming style per class suggests the classes came from different sources.
```

## VERDICT

```
FAIL - 5 issue(s):
   1. [WARN] Only 475 images - expect unstable results; report mean +/- std over several seeds, not a single number.
   2. [FAIL] 67% of images sit in single-class size blocks; dimensions alone score 74% balanced accuracy.
   3. [FAIL] 4 photo(s) appear in more than one split - the test score would be inflated. Re-run src/split_data.py.
   4. [FAIL] 1 file(s) training cannot read - delete or re-split.
   5. [WARN] 93 images are smaller than 224px and will be blurry when upscaled (by class: {'flooding': 0, 'normal': 93}).
```
