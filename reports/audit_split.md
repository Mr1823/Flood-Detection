# Dataset audit

| | |
|---|---|
| Dataset | `data` |
| Images audited | 1525 (all images) |
| Seed | 42 |
| Audited | 2026-10-04T19:59:59 |
| Verdict | **FAIL** |
| Fingerprint | `5a3a30669ccf26d1...` |

## DATASET AUDIT: data (split)

```
1525 images (all images) | 2 classes: flood, no_flood | splits: test, train, val
```

## 1. COUNTS

```
   test: flood=78  no_flood=150
   train: flood=368  no_flood=700
   val: flood=79  no_flood=150
   class ratio 1.9:1
```

## 2. SIZE LEAK (original dimensions)

```
   balanced accuracy from width x height alone :  55.9%   (chance 50.0%)
   plain accuracy                              :  69.1%   (always 'no_flood': 65.6%)
      flood                    462 distinct sizes
      no_flood                  91 distinct sizes
   images in a large single-class size block   :  53.2%
        237 images at 399x360 -> 100% 'no_flood'
        193 images at 400x360 -> 100% 'no_flood'
        183 images at 640x360 -> 96% 'no_flood'
   FAIL: image size gives the label away (56% balanced from size alone, 53% of images in single-class size blocks).
```

## 3. DUPLICATES

```
   exact copies (MD5)                     : 0
   near copies (256-bit hash, <= 25 bits) : 0
   copies across splits                   : 0  ok
   copies across classes                  : 0  ok
```

## 4. IMAGE QUALITY

```
   unreadable files                 : 0
   formats TensorFlow cannot load   : 0
   smaller than 224px              : 34 (2%)  by class: {'flood': 17, 'no_flood': 17}
   formats: {'JPEG': 1525}
   colour modes: {'RGB': 1525}
```

## 4b. SOURCE LEAK (content-free statistics after framing to 224x224)

```
   A random forest sees only blur, contrast and colour spread (chance = 50.0%,
   balanced accuracy, 5-fold cross-validation):
      whole image squashed to 224x224 (as in the earlier audits)
         1525 images:  68.1%   (blur 54.4%, contrast 55.1%, colour_spread 56.4%)
      centred 3:4-4:3 crop, then 224x224 (what the model receives)
         1525 images:  69.5%   (blur 52.0%, contrast 55.2%, colour_spread 54.8%)
   No pass line: the score is reported, not judged. It is a FLOOR on the
   shortcut, not a measurement of it - a fine-tuned CNN exploits the same
   signal far more effectively than three numbers and a random forest.
```

## 5. FILENAME PATTERNS

```
   flood                  e.g. flood_image0005.jpg                      prefix 'flood_' on 100%
   no_flood               e.g. normal_image0029.jpg                     prefix 'normal' on 100%
   -> one naming style per class suggests the classes came from different sources.
```

## VERDICT

```
FAIL - 1 issue(s):
   1. [FAIL] 53% of images sit in single-class size blocks; dimensions alone score 56% balanced accuracy.
```
