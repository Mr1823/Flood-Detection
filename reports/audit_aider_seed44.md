# Dataset audit

| | |
|---|---|
| Dataset | `aider_raw/AIDER` |
| Images audited | 1052 (class-balanced random sample, seed 44, from flood 526, no_flood 4390) |
| Seed | 44 |
| Audited | 2026-10-04T20:47:50 |
| Verdict | **FAIL** |
| Fingerprint | `fab669e9d86d186f...` |

## DATASET AUDIT: aider_raw/AIDER (pool)

```
1052 images (class-balanced random sample, seed 44, from flood 526, no_flood 4390) | 2 classes: flood, no_flood | splits: all
```

## 1. COUNTS

```
   all: flood=526  no_flood=526
   class ratio 1.0:1
```

## 2. SIZE LEAK (original dimensions)

```
   balanced accuracy from width x height alone :  94.5%   (chance 50.0%)
   plain accuracy                              :  94.5%   (always 'flood': 50.0%)
      flood                    462 distinct sizes
      no_flood                  46 distinct sizes
   images in a large single-class size block   :  26.5%
        125 images at 399x360 -> 100% 'no_flood'
        118 images at 400x360 -> 100% 'no_flood'
         13 images at 512x512 -> 100% 'no_flood'
   FAIL: image size gives the label away (94% balanced from size alone, 27% of images in single-class size blocks).
```

## 3. DUPLICATES

```
   exact copies (MD5)                     : 0
   near copies (256-bit hash, <= 25 bits) : 5
      all/flood/flood_image0098.jpg  ~=  all/flood/flood_image0466.jpg  (6/256 bits differ)
      all/no_flood/normal_image0433.jpg  ~=  all/no_flood/normal_image3197.jpg  (23/256 bits differ)
      all/no_flood/normal_image3001.jpg  ~=  all/no_flood/normal_image3003.jpg  (25/256 bits differ)
      all/no_flood/normal_image3655.jpg  ~=  all/no_flood/normal_image3659.jpg  (19/256 bits differ)
      all/no_flood/normal_image3915.jpg  ~=  all/no_flood/normal_image3919.jpg  (13/256 bits differ)
   copies across classes                  : 0  ok
   (split_data.py keeps one copy of each before splitting)
```

## 4. IMAGE QUALITY

```
   unreadable files                 : 0
   formats TensorFlow cannot load   : 0
   smaller than 224px              : 29 (3%)  by class: {'flood': 17, 'no_flood': 12}
   formats: {'JPEG': 1052}
   colour modes: {'RGB': 1052}
```

## 4b. SOURCE LEAK (content-free statistics after framing to 224x224)

```
   A random forest sees only blur, contrast and colour spread (chance = 50.0%,
   balanced accuracy, 5-fold cross-validation):
      whole image squashed to 224x224 (as in the earlier audits)
         1052 images:  66.0%   (blur 55.5%, contrast 55.7%, colour_spread 56.9%)
      centred 3:4-4:3 crop, then 224x224 (what the model receives)
         1052 images:  65.7%   (blur 52.7%, contrast 54.8%, colour_spread 57.4%)
   No pass line: the score is reported, not judged. It is a FLOOR on the
   shortcut, not a measurement of it - a fine-tuned CNN exploits the same
   signal far more effectively than three numbers and a random forest.
```

## 5. FILENAME PATTERNS

```
   flood                  e.g. flood_image0001.jpg                      prefix 'flood_' on 100%
   no_flood               e.g. normal_image0006.jpg                     prefix 'normal' on 100%
   -> one naming style per class suggests the classes came from different sources.
```

## VERDICT

```
FAIL - 1 issue(s):
   1. [FAIL] 27% of images sit in single-class size blocks; dimensions alone score 94% balanced accuracy.
```
