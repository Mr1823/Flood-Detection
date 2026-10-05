# Flood Detection System — MobileNetV2 transfer learning

Binary image classification of **aerial images**: **flood** vs **no_flood**.
MobileNetV2 with ImageNet weights is first trained as a frozen feature extractor
and then fine-tuned. The project delivers a trained model, an evaluation report,
Grad-CAM evidence for every decision, a React dashboard with live prediction and
TFLite exports with measured size and speed.

Every result in this README comes from an actual run; each table names the file
in `reports/` that holds its numbers. Nothing is estimated or typed in.

## Results at a glance

Deployed model: fine-tuned MobileNetV2 (seed 43), decision threshold **0.30**
chosen on the validation set. Test set: 228 images (78 flood, 150 no_flood).

| Test set, threshold 0.30 | Value |
|---|---|
| Recall (floods found) | 0.987 — 77 of 78 |
| Precision | 0.865 — 12 false alarms among 150 non-floods |
| F1 | 0.922 |
| ROC AUC | 0.9955 |
| Average precision (PR AUC) | 0.9932 |

Source: `reports/metrics.json`. Across three seeds the fine-tuned variant reaches
test accuracy 0.9649 ± 0.0044 at threshold 0.5 (`reports/seed_summary.json`).

## Results dashboard (React, localhost)

The project's user interface, and a read-only analytics view over the numbers already
in `reports/`. Five of its six tabs are static and only ever show values the pipeline
has already produced; the **Predict** tab runs the model through a small local API.

```bash
cd frontend
npm install            # once
npm run data           # regenerate frontend/public/data/dashboard.json from reports/
npm run dev            # http://localhost:5173
```

`npm run data` runs `backend/src/export_dashboard_data.py`, so refreshing the dashboard after a
retrain is one command. The exporter fails loudly if a source report is missing and
names the script that produces it.

### Live prediction in the dashboard

Five of the six tabs are static and need nothing running. The **Predict** tab is the
exception: it uploads an image to a small local API that runs the deployed model and
returns the probability and a Grad-CAM overlay. Start it in a second terminal:

```bash
cd frontend
npm run api            # http://127.0.0.1:8000, proxied as /api by the dev server
```

`backend/src/serve_api.py` reuses `dataset.load_for_model`, `gradcam.explain` and
`model.load_trained_model` - the same functions `evaluate.py` and `predict.py` use, so
a prediction in the dashboard is the same computation as one in the pipeline. Checked
against the saved pipeline values, the API agrees to 7e-07.

The **Threshold** tab also has a slider that recomputes precision, recall, the
confusion matrix, floods missed and false alarms from the saved per-image test
probabilities. That one needs no API: it re-counts saved numbers in the browser and is
exact. The deployed threshold is unchanged by it.

| Script | What it does |
|---|---|
| `npm run api` | local prediction service for the Predict tab |
| `npm run dev:all` | the API and the dev server together |

| Script | What it does |
|---|---|
| `npm run data` | rebuild `dashboard.json` from `reports/` |
| `npm run dev` | dev server on http://localhost:5173 |
| `npm run build` | production build into `frontend/dist` |
| `npm run preview` | serve that build locally |

Two of the dashboard's inputs are written by `backend/src/evaluate.py`: `reports/curves.json`
(ROC and PR curve points) and `reports/test_probabilities.csv` (P(flood) per test
image). Re-run `python backend/src/evaluate.py` if either is missing.


## Setup (macOS, Apple Silicon)

```bash
brew install python@3.11 python-tk@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt     # pinned, tested set - incl. protobuf==4.25.9, see below
```

Every script prints the TensorFlow version and the devices it can see; `GPU`
means the Metal GPU (tensorflow-metal) is in use. Run everything from the project
folder, inside `.venv`.

## Dataset

**AIDER** — *Aerial Image Dataset for Emergency Response applications*, Christos
Kyrkou, Zenodo 2020, DOI [10.5281/zenodo.3888300](https://doi.org/10.5281/zenodo.3888300),
licensed **CC BY 4.0** (use with attribution). Please also cite the two papers it
accompanies: DOI 10.1109/JSTARS.2020.2969809 and DOI 10.1109/CVPRW.2019.00077.

> The raw archive and its `aider_raw/` extract are **not kept in this repo** - only the
> finished splits in `data/` are. Run the download below before re-running
> `split_data.py` or auditing the source dataset; everything else (training,
> evaluation, Grad-CAM, export, the app and the dashboard) works from `data/` alone.

```bash
curl -L -C - -o AIDER.zip "https://zenodo.org/records/3888300/files/AIDER.zip?download=1"
md5 AIDER.zip                        # must print 1ad4eb02ed156e8dfa19986ff382e58b
mkdir -p aider_raw && unzip -q AIDER.zip -d aider_raw    # -> aider_raw/AIDER/<class>/
```

Zenodo throttles each connection (about 35 KB/s was observed), so the 263 MB
download can take two hours; `-C -` resumes an interrupted download.

Only two AIDER classes are used: `flooded_areas` (526 images) → **flood** and
`normal` (4,390 images) → **no_flood**. `backend/src/split_data.py` then:

1. skips files that cannot be used — here one all-black video frame;
2. removes duplicates over the **whole** pool, before any subsampling: 0 exact
   copies (MD5) and 200 near-copies (256-bit difference hash; 199 no_flood,
   1 flood) — the ones inspected were consecutive frames of the same drone video;
3. subsamples no_flood from 4,190 to **1,000** images with `config.SEED`
   (`reports/subsample_manifest.csv` lists them); flood keeps all 525;
4. splits each class 70/15/15:

| Split | no_flood | flood | total |
|---|---|---|---|
| train | 700 | 368 | 1,068 |
| val | 150 | 79 | 229 |
| test | 150 | 78 | 228 |

`data/split_manifest.csv` records where every source file went, or why it was
left out. Training uses class weights 0.763 (no_flood) and 1.451 (flood).

## Running, in order

```bash
python backend/src/audit_dataset.py --source --balance   # 1. audit AIDER for shortcuts -> reports/audit_aider.md
python backend/src/split_data.py                          # 2. de-duplicate, subsample, split; audits the split
python backend/src/dataset.py                             # 3. shapes, label mapping, sanity-check batch
python backend/src/train.py                               # 4. both variants x 3 seeds (~15 min on an M4)
python backend/src/evaluate.py                            # 5. threshold chosen on validation, test report
python backend/src/gradcam.py                             # 6. Grad-CAM grid, attention + deletion test
python backend/src/export.py                              # 7. .keras + TFLite exports, sizes and latencies
python backend/src/predict.py path/to/image_or_folder     # command-line predictions
```

Every script has `--help`; paths and the main hyperparameters can be overridden,
and every constant lives in `backend/src/config.py`.

**The app** is the React dashboard (see *Results dashboard* above). Its **Predict**
tab takes an image and shows it next to its Grad-CAM evidence, the verdict, the flood
probability, the decision threshold and the attention area. Its **Threshold** tab has
a slider that starts at the tuned threshold and recomputes precision, recall and the
confusion matrix live as it moves. Unreadable files are reported, not crashed on.

## Method

- **Input framing (the shortcut mitigation, see *Dataset validation*).** Training
  images are cut with a *random-resized crop* — a random window covering 70–100 %
  of the image, aspect ratio 3:4 to 4:3 — and resized to 224×224, so no two epochs
  see the same framing. A window that does not fit is shrunk, never stretched.
  Validation, test and every app use a deterministic centre crop with the same
  aspect range.
- **Augmentation**, inside the model and active only during training:
  horizontal flip, rotation ±18° (`RandomRotation(0.05)`), brightness ±20 %,
  contrast ±20 %. There is no RandomZoom: the random crop already varies scale.
- **Preprocessing.** `mobilenet_v2.preprocess_input` (0..255 → −1..1) is a layer
  inside the model, after augmentation — the only place it happens, so no caller
  can forget it or apply it twice.
- **Model.** MobileNetV2 (ImageNet) built with `input_tensor`, so `Conv_1` is
  reachable for Grad-CAM → global average pooling → Dropout 0.3 → Dense 128 ReLU →
  Dropout 0.2 → Dense 1 sigmoid.
- **Two variants.** *Without fine-tuning*: backbone frozen, Adam 1e-3, up to 15
  epochs. *With fine-tuning*: continues from those weights, unfreezes from
  `block_13_expand` (BatchNorm stays frozen), recompiles with Adam 1e-5, up to 15
  more epochs. Early stopping on validation AUC (patience 5) and LR reduction on
  validation loss.
- **Selection uses validation data only.** The variant with the higher mean
  validation AUC over three seeds is selected; the deployed weights are that
  variant's run with the highest validation AUC. The decision threshold is also
  chosen on validation. The test set never influences a choice.

## Results

### Model comparison — 3 seeds (`reports/seed_summary.json`)

Mean ± sample standard deviation over seeds 42, 43, 44. Accuracy, precision and
recall use the default threshold 0.5; AUC here is Keras' 200-threshold estimate.

| Model | Val loss | Val accuracy | Val AUC | Test accuracy |
|---|---|---|---|---|
| Without fine-tuning | 0.1375 ± 0.1017 | 0.9389 ± 0.0492 | 0.9966 ± 0.0010 | 0.9474 ± 0.0494 |
| With fine-tuning | 0.0662 ± 0.0057 | 0.9738 ± 0.0044 | 0.9982 ± 0.0009 | 0.9649 ± 0.0044 |

| Test set, threshold 0.5 | Accuracy | AUC | Precision | Recall |
|---|---|---|---|---|
| Without fine-tuning | 0.9474 ± 0.0494 | 0.9894 ± 0.0004 | 0.8878 ± 0.1092 | 0.9829 ± 0.0074 |
| With fine-tuning | 0.9649 ± 0.0044 | 0.9934 ± 0.0028 | 0.9309 ± 0.0225 | 0.9701 ± 0.0148 |

**Selected: with fine-tuning.** The frozen variant's large spread comes from one
seed: seed 42's frozen run stopped early (8 epochs, best epoch 3) and at
threshold 0.5 reached test precision 0.762, against 0.962 and 0.939 for the other
seeds, while its AUC (0.989) matched theirs — it ranks images as well, but 0.5 is
a poor cut-off for it. Per seed, seed 44 preferred the frozen variant (validation
AUC 0.9977 vs 0.9976); across seeds fine-tuning wins.

### Deployed model on the test set (`reports/metrics.json`)

**Threshold.** The rule: the lowest threshold whose *validation* precision is
still ≥ 0.85, which maximises recall under that floor — a missed flood costs
lives, a false alarm costs an inspection. It selected **0.30** (validation
precision 0.859, recall 1.000). Both sweeps are saved:
`reports/threshold_sweep_val.csv` (used to choose) and
`reports/threshold_sweep_test.csv` (reported only). The dashboard's **Threshold**
tab draws both.

| Confusion matrix, test (rows = actual, columns = predicted) | pred. no_flood | pred. flood |
|---|---|---|
| actual no_flood — threshold 0.50 | 142 | 8 |
| actual flood — threshold 0.50 | 1 | 77 |
| actual no_flood — threshold 0.30 | 138 | 12 |
| actual flood — threshold 0.30 | 1 | 77 |

| Test set | Threshold 0.50 | Threshold 0.30 (tuned) |
|---|---|---|
| Precision | 0.906 | 0.865 |
| Recall | 0.987 | 0.987 |
| F1 | 0.945 | 0.922 |
| Accuracy | 0.961 | 0.943 |
| Specificity | 0.947 | 0.920 |

ROC AUC 0.9955, average precision 0.9932 (`reports/curves.json`; the dashboard's
**Performance** tab draws both curves).

**A note on the threshold rule.** On validation, recall is 1.000 at every swept
threshold from 0.05 to 0.45, so "lowest threshold" gives up precision there for
no validation recall. Breaking that tie towards the highest such threshold would
pick 0.45 (validation precision 0.919, recall 1.000; on test: precision 0.895,
recall 0.987). The rule was kept as specified. For comparison, recall ≥ 0.95 on
validation is reached up to threshold 0.90 (validation precision 1.000, recall
0.975), which on test gives precision 0.961 but recall 0.949 — 4 missed floods.

**The missed flood** (`reports/false_negatives.csv`): one of 78 —
`flood_image0384.jpg`, P(flood) = 0.0139. It is a straight-down, map-style
aerial photograph of a city with brown water in the river and the streets
(captioned "nearmap", "13 Jan 2011"). Most training flood photos are oblique
views, while straight-down urban views appear mostly among the no_flood images,
so this miss is consistent with the viewpoint and source differences the
dataset audit measured.

**False alarms** (12 at threshold 0.30, `reports/metrics.json`) are mostly
open water (a turquoise sea with swimmers), brown earth or mud (a motocross
track, a beach, a muddy field) and vehicles on roads: water and mud colours act
as flood evidence even when there is no flood.

## Grad-CAM evidence (`reports/gradcam_stats.json`)

Grad-CAM on `Conv_1` (7×7 feature maps), computed from the logit before the
sigmoid (the same map up to scale, but immune to the sigmoid saturating at
P = 1.0). *Attention area* = share of the image whose normalised heat exceeds 0.5.

| Test images | n | Mean attention area | Share above 50 % |
|---|---|---|---|
| All | 228 | 26.2 % | 26.3 % |
| True flood | 78 | 62.0 % | 76.9 % |
| True no_flood | 150 | 7.5 % | 0.0 % |

The low overall figure comes from the non-flood images, which contain little
flood evidence. For actual floods the evidence covers most of the frame, and
three in four exceed the 50 % warning line. In AIDER's aerial flood photos the
water itself often fills most of the frame, so a large area is expected whether
the model looks at the water or at the whole scene: **attention area measures
how spread the evidence is, not where it is**, and on its own it does not settle
the question.

A size-controlled **deletion test** does: on the 77 detected test floods, grey
out the 25 % of pixels Grad-CAM ranks highest, and separately the 25 % it ranks
lowest.

| Detected floods (77) | Mean P(flood) | Still classed flood |
|---|---|---|
| Original | 0.992 | 100 % |
| Most-attended 25 % removed | 0.848 | 88.3 % |
| Least-attended 25 % removed | 0.987 | 100.0 % |

Removing what Grad-CAM points at hurts far more than removing what it ignores,
so the evidence is concentrated where the maps say — not spread evenly over the
scene. It is also redundant: most floods survive losing their most-attended
quarter, which fits water covering much of these frames. The grid shows the heat
on flooded streets and water — and, in the false alarms, on open sea, brown earth
and vehicles. What the test cannot show is whether the evidence is the water
itself rather than something correlated with how the photos were taken.

## Deployment: export, size and speed (`reports/export_benchmark.json`)

Measured on an Apple M4: mean ± std of 50 single-image runs after 5 untimed
warm-up runs; agreement checked on all 228 test images against the Keras model
on the CPU.

| File | Size | Latency | Device | max \|ΔP\| vs Keras | Decisions changed at 0.30 | Test precision / recall at 0.30 |
|---|---|---|---|---|---|---|
| `models/flood_mobilenetv2.keras` | 9.81 MB | 14.13 ± 0.98 ms | Metal GPU | 0.000 | 0 | 0.865 / 0.987 |
| `models/flood_mobilenetv2.tflite` (8-bit weights) | 2.54 MB | 4.39 ± 0.03 ms | CPU | 0.237 | 2 of 228 | 0.885 / 0.987 |
| `models/flood_mobilenetv2_fp16.tflite` | 4.57 MB | 7.34 ± 0.03 ms | CPU | 0.033 | 0 | 0.865 / 0.987 |

The float16 TFLite file is under half the size of the Keras file, runs in
7.34 ms on a CPU and reproduces every Keras decision on the test set — the safe
choice for a phone or a drone. The 8-bit file is the smallest and fastest; it
changed two decisions, both Keras false alarms that it scored below 0.30, but its
probabilities move by up to 0.237, so its threshold should be re-validated before
it is deployed. (Keras ran on the GPU and TFLite on the CPU, so the latencies
compare deployment options, not file formats alone.)

## Dataset validation

Before training on any dataset, `backend/src/audit_dataset.py` checks whether something
other than flood water separates the two classes. A CNN learns such a shortcut
more easily than the content, and the test score then measures the shortcut.
Its key test trains a random forest on three statistics that say nothing about
what an image shows — blur, contrast and colour spread, measured after the
224×224 framing — and reports the balanced accuracy (chance = 50 %). **That score
is a floor on the shortcut, not a measurement of it**: a fine-tuned CNN with
2.2 million parameters exploits the same signal far more effectively than three
hand-made numbers and a small random forest.

### The first dataset was rejected (`reports/audit_rejected.md`)

A download of 476 files, pre-split into `train/valid/test` with `flooding/` and
`normal/` folders.

- **100 % dimension leak.** All 253 flood images — and no non-flood image — were
  exactly 512×384, while the 222 non-flood images came in 136 sizes. The rule
  "512×384 → flood, anything else → no_flood" would label every image correctly
  without looking at it. (The audit's own size test, a leave-one-out lookup by
  exact size, scored 73.6 % balanced accuracy; 67.4 % of all images sat in
  single-class size blocks.)
- **Source leak.** The content-free statistics classified the images at
  **74.6 %** balanced accuracy; blur alone reached 66.7 %. All 93 images smaller
  than 224 px were non-flood, so only that class was blurred by upscaling.
- **Contamination.** 4 photos appeared, resized, in more than one of the
  download's own splits.

**Why cleaning did not fix it.** The difference lay in how each class was
produced — apparently one curated source for every flood photo, web scraping for
the rest — not in a few bad files. Removing all 93 images under 224 px, which
removes every icon and thumbnail, **raised** the content-free score to 77.6 %
(blur alone 68.8 %) and the share of images in single-class size blocks from
67.4 % to 70.3 % (`reports/audit_rejected_cleaned.md`). Cleaning shrinks the
scraped class; the flood class keeps its single-source signature.

### Corrections made to the audit itself

- An early version also used each image's original width and height. The model
  never receives those numbers, and they inflated the leak to 99.8 %; the test now
  uses only statistics measured after the 224×224 framing. (An exploratory run on
  the de-duplicated split gave 69.7 % plain accuracy against a 53.9 % baseline;
  every such number is a lower bound.)
- A 64-bit near-duplicate hash with a 5-bit threshold flagged different images
  as copies (a lightbulb icon and a flooded road). The audit now uses a 256-bit
  difference hash with a 25-bit cut-off: real copies differed by 0–16 bits, the
  closest different pair by 33.
- All audit accuracies are balanced, so chance is 50 % at any class ratio.

### Why AIDER, and what its audit shows

AIDER is a published research dataset with an open licence, and its flood and
normal classes are both aerial views — no icons, document scans or product
photos. It is not free of source differences, and they are reported here:

| Audit | Images | Content-free score | Report |
|---|---|---|---|
| Full pool (8.3 : 1 no_flood : flood) | 4,916 | 61.9 % | `audit_aider_full.md` |
| Balanced sample, seed 42 | 1,052 | 69.7 % | `audit_aider.md` |
| Balanced sample, seed 43 | 1,052 | 67.7 % | `audit_aider_seed43.md` |
| Balanced sample, seed 44 | 1,052 | 66.0 % | `audit_aider_seed44.md` |
| Balanced sample, seed 45 | 1,052 | 67.2 % | `audit_aider_seed45.md` |
| Final split, squashed to 224×224 | 1,525 | 68.1 % | `audit_split.md` |
| Final split, centred crop the model sees | 1,525 | 69.5 % | `audit_split.md` |

The 8:1 pool understated the leak; balanced samples put it at 66–70 %. No single
statistic separates the classes — on the final split none exceeds 56.4 % on its
own; it is their combination. Image sizes tell the same story: much of the
no_flood class comes from a few fixed-size video-frame pipelines — in the final
split 237 images at 399×360, 193 at 400×360 and 183 at 640×360, almost all
no_flood — while the flood photos come in 462 sizes. A 65 % pass line was first
proposed for this score, then withdrawn as arbitrary: a score of 66–70 % was
accepted as typical for scraped imagery and is reported, not enforced. The audit
now reports and never blocks training.

### The mitigation: random-resized crop

Squashing every image to a square preserves the uniform capture geometry that
the audit detected — every 400×360 frame is distorted in exactly the same way,
epoch after epoch. Random cropping means no two epochs see the same framing, so
that geometry stops being a stable signal the model can lean on.

What it cannot change: blur, contrast and colour statistics. With the centred
crop the model sees, the content-free score is 69.5 % — unchanged — because those
statistics do not depend on framing. The evidence on what the model does use is
in *Grad-CAM evidence* above: its flood evidence is concentrated where Grad-CAM
points (the deletion test), the heat sits on flooded streets and water, and the
errors show water and mud colours being over-trusted. Whether part of the
evidence is still source-related cannot be ruled out with these tests.

## Engineering findings

These were found and measured while building the project.

- **tensorflow-metal computed wrong results.** The plugin's own graph optimiser
  corrupted the `Dense(128)` layer in graph mode — the mode `fit`, `evaluate` and
  `predict` use: in a test model, one image scored P(flood) 0.258 on the Metal GPU
  and 0.565 on the CPU. Comparing every layer, GPU graph against CPU, located it
  (agreement to about 1e-6 up to global pooling, then a relative error of about
  160 % at the Dense layer). `config.py` now disables only the plugin optimiser,
  and `model.check_device_consistency()` re-checks GPU against CPU before training,
  on every saved model and before evaluation. The first full training run used the
  broken graph and was discarded; every result here comes from the corrected run.
  Epochs also became faster (9 s instead of 12 s).
- **Keras 3 exported the model with augmentation switched on.** `model.export()`
  traced the call with the training flag left at its default, so the exported
  graph randomly rotated and re-brightened every image. `backend/src/export.py` exports an
  explicit `training=False` endpoint instead.
- **Protobuf has to be pinned.** TensorFlow 2.16 cannot import protobuf 5.x or
  newer, and several web packages will happily upgrade it as a transitive
  dependency; `requirements.txt` pins `protobuf==4.25.9`.
- **Integer random draws fail on the Metal GPU**, so the random crop draws floats
  in [0, 1) and scales them.

## Known limitations

- **Aerial imagery only.** AIDER contains drone and aircraft views, so
  performance on ground-level photos — phones, street CCTV — is untested.
- **no_flood is subsampled.** Training and testing used 1,000 of the 4,190
  de-duplicated non-flood images (about 1.9 : 1 against flood; AIDER itself is
  8.3 : 1), so the deployed class balance differs from training. Where non-flood
  scenes are far more common, the same false-positive rate produces more false
  alarms per real flood, and the measured precision does not carry over; the
  threshold should be re-checked on data with the deployment's own class balance.
- **Source differences remain.** The content-free score is 66–70 % on AIDER, and
  it is a lower bound. The crop mitigation addresses capture geometry, not colour
  or sharpness statistics.
- **Small test set.** 228 test images (78 floods) from one split; a single missed
  flood moves recall by about 1.3 points. Seed-to-seed spread is reported.
- **Near-copies beyond the cut-off.** Video frames of the same scene that differ
  by more than 25 of 256 hash bits can still sit in different splits.
- **Water is not flood.** False alarms show open water and brown earth being
  treated as flood evidence.
- **Unseen geographies are untested.** Performance on regions, building styles
  and flood types absent from AIDER is unknown.
- **Grad-CAM is coarse** — a 7×7 map upsampled to 224×224.
# Flood-Detection
