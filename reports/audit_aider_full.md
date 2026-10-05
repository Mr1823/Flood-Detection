# Dataset audit

| | |
|---|---|
| Dataset | `aider_raw/AIDER` |
| Images audited | 4916 (all images) |
| Seed | 42 |
| Audited | 2026-10-04T20:46:50 |
| Verdict | **FAIL** |
| Fingerprint | `d83d9ba230a4a4e7...` |

## DATASET AUDIT: aider_raw/AIDER (pool)

```
4916 images (all images) | 2 classes: flood, no_flood | splits: all
```

## 1. COUNTS

```
   all: flood=526  no_flood=4390
   class ratio 8.3:1
```

## 2. SIZE LEAK (original dimensions)

```
   balanced accuracy from width x height alone :  55.2%   (chance 50.0%)
   plain accuracy                              :  90.0%   (always 'no_flood': 89.3%)
      flood                    462 distinct sizes
      no_flood                 315 distinct sizes
   images in a large single-class size block   :  54.5%
        960 images at 400x360 -> 100% 'no_flood'
        957 images at 399x360 -> 100% 'no_flood'
        522 images at 240x240 -> 99% 'no_flood'
   FAIL: image size gives the label away (55% balanced from size alone, 54% of images in single-class size blocks).
```

## 3. DUPLICATES

```
   exact copies (MD5)                     : 0
   near copies (256-bit hash, <= 25 bits) : 249
      all/flood/flood_image0098.jpg  ~=  all/flood/flood_image0466.jpg  (6/256 bits differ)
      all/no_flood/normal_image0004.jpg  ~=  all/no_flood/normal_image0126.jpg  (12/256 bits differ)
      all/no_flood/normal_image0004.jpg  ~=  all/no_flood/normal_image0248.jpg  (18/256 bits differ)
      all/no_flood/normal_image0004.jpg  ~=  all/no_flood/normal_image4270.jpg  (11/256 bits differ)
      all/no_flood/normal_image0010.jpg  ~=  all/no_flood/normal_image0012.jpg  (24/256 bits differ)
   copies across classes                  : 0  ok
   (split_data.py keeps one copy of each before splitting)
```

## 4. IMAGE QUALITY

```
   unreadable files                 : 0
   formats TensorFlow cannot load   : 0
   smaller than 224px              : 112 (2%)  by class: {'flood': 17, 'no_flood': 95}
   formats: {'JPEG': 4916}
   colour modes: {'RGB': 4916}
```

## 4b. SOURCE LEAK (content-free statistics after framing to 224x224)

```
   A random forest sees only blur, contrast and colour spread (chance = 50.0%,
   balanced accuracy, 5-fold cross-validation):
      whole image squashed to 224x224 (as in the earlier audits)
         2000 images:  61.9%   (blur 51.7%, contrast 58.9%, colour_spread 52.7%)
      centred 3:4-4:3 crop, then 224x224 (what the model receives)
         2000 images:  59.7%   (blur 49.3%, contrast 57.5%, colour_spread 58.5%)
   No pass line: the score is reported, not judged. It is a FLOOR on the
   shortcut, not a measurement of it - a fine-tuned CNN exploits the same
   signal far more effectively than three numbers and a random forest.
```

## 5. FILENAME PATTERNS

```
   flood                  e.g. flood_image0001.jpg                      prefix 'flood_' on 100%
   no_flood               e.g. normal_image0001.jpg                     prefix 'normal' on 100%
   -> one naming style per class suggests the classes came from different sources.
```

## VERDICT

```
FAIL - 2 issue(s):
   1. [WARN] Classes are imbalanced 8.3:1 - train with class weights and judge by AUC / balanced metrics, not plain accuracy.
   2. [FAIL] 54% of images sit in single-class size blocks; dimensions alone score 55% balanced accuracy.
```
