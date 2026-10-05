# Dataset audit

| | |
|---|---|
| Dataset | `/private/tmp/claude-501/-Users-blackheart-Projects--Department-Project--DP-Mini-Project-/e1c42629-47c6-4164-bf67-7f78e6a3017f/scratchpad/rejected_cleaned_v2` |
| Images audited | 381 (all images) |
| Seed | 42 |
| Audited | 2026-10-04T20:46:33 |
| Verdict | **FAIL** |
| Fingerprint | `c81c95efe358204c...` |

## DATASET AUDIT: /private/tmp/claude-501/-Users-blackheart-Projects--Department-Project--DP-Mini-Project-/e1c42629-47c6-4164-bf67-7f78e6a3017f/scratchpad/rejected_cleaned_v2 (split)

```
381 images (all images) | 2 classes: flooding, normal | splits: test, train, valid
```

## 1. COUNTS

```
   test: flooding=43  normal=19
   train: flooding=168  normal=89
   valid: flooding=42  normal=20
   class ratio 2.0:1
```

## 2. SIZE LEAK (original dimensions)

```
   balanced accuracy from width x height alone :  64.1%   (chance 50.0%)
   plain accuracy                              :  75.9%   (always 'flooding': 66.4%)
      flooding                   1 distinct sizes
      normal                   102 distinct sizes
   images in a large single-class size block   :  70.3%
        253 images at 512x384 -> 100% 'flooding'
          5 images at 1300x956 -> 100% 'normal'
          5 images at 1280x720 -> 100% 'normal'
   FAIL: image size gives the label away (64% balanced from size alone, 70% of images in single-class size blocks).
```

## 3. DUPLICATES

```
   exact copies (MD5)                     : 0
   near copies (256-bit hash, <= 25 bits) : 3
      test/normal/stock-photo-red-dirt-road-rural-roads-in  ~=  train/normal/images (3).jpg  (16/256 bits differ)
      test/normal/village-roads (1).jpg  ~=  valid/normal/images (2).jpg  (16/256 bits differ)
      train/normal/drawing-line-suburban-roads (1).jpg  ~=  train/normal/drawing-line-suburban-roads.jpg  (14/256 bits differ)
   copies across splits                   : 2  <- test set is contaminated
   copies across classes                  : 0  ok
```

## 4. IMAGE QUALITY

```
   unreadable files                 : 0
   formats TensorFlow cannot load   : 0
   smaller than 224px              : 0 (0%)  by class: {'flooding': 0, 'normal': 0}
   formats: {'JPEG': 371, 'PNG': 9, 'GIF': 1}
   colour modes: {'RGB': 370, 'P': 9, 'RGBA': 1, 'L': 1}  -> dataset.py decodes all as 3-channel RGB
```

## 4b. SOURCE LEAK (content-free statistics after framing to 224x224)

```
   A random forest sees only blur, contrast and colour spread (chance = 50.0%,
   balanced accuracy, 5-fold cross-validation):
      whole image squashed to 224x224 (as in the earlier audits)
         381 images:  77.6%   (blur 68.8%, contrast 53.2%, colour_spread 54.6%)
      centred 3:4-4:3 crop, then 224x224 (what the model receives)
         381 images:  78.2%   (blur 66.0%, contrast 48.7%, colour_spread 51.8%)
   No pass line: the score is reported, not judged. It is a FLOOR on the
   shortcut, not a measurement of it - a fine-tuned CNN exploits the same
   signal far more effectively than three numbers and a random forest.
```

## 5. FILENAME PATTERNS

```
   flooding               e.g. image_211.jpg                            prefix 'image_' on 100%
   normal                 e.g. Sampling-area-urban-suburban-rural-loc   prefix 'ia_100' on 41%
   -> one naming style per class suggests the classes came from different sources.
```

## VERDICT

```
FAIL - 3 issue(s):
   1. [WARN] Only 381 images - expect unstable results; report mean +/- std over several seeds, not a single number.
   2. [FAIL] 70% of images sit in single-class size blocks; dimensions alone score 64% balanced accuracy.
   3. [FAIL] 2 photo(s) appear in more than one split - the test score would be inflated. Re-run src/split_data.py.
```
