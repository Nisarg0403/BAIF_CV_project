# Master Project Plan
## Computer Vision-Based Body Weight Estimation of Dairy Cattle
### A Deep Learning Approach for Indian Smallholder Dairy Systems

**Domain:** Agriculture & AgriTech — Livestock Management
**Prepared for:** BAIF Development Research Foundation
**Author:** Nisarg Vakharia, FY MSc – AI/ML
**Faculty Guide:** Dr. Jayashree Prasad

> **Note on scope reconciliation:** an alternative plan was reviewed that assumes a four-photo (front/rear/left/right) capture protocol and a pre-existing large manual-measurement dataset. That plan's *methodology* (data-splitting discipline, baseline experiment structure, terminology precision, bad-input handling) has been merged into this document. Its *four-view, multi-photo capture design* has **not** been adopted for the current phase, since it conflicts with the single side-view protocol already agreed and executed with BAIF (field visit, 27 Aug 2026). Four-view capture is retained as documented **future work** (Section 14).

---

## 1. Project Overview

### Core Idea
A field worker photographs a dairy animal with a smartphone (single side-view image, current phase), and the system automatically detects the animal, segments it from the background, extracts body measurements, and predicts live body weight — without a weighing scale, manual measurement, or physical contact.

The design principle, carried over from reviewed methodology: **do not make the model guess weight from pixels alone when the pipeline can first estimate meaningful physical body features.**

```
Image → Detection → Segmentation → Body Measurements → Calibration → ML Regression → Weight
```

This is an **explainable pipeline**, not a black-box image-to-kilograms model — every intermediate output (silhouette, measurements) can be inspected and validated against manual measurements.

---

## 2. Objectives

The system should:
1. Accept a single smartphone image of the animal.
2. Detect the animal in the image.
3. Segment the animal from the background.
4. Estimate physical body measurements (length, height, girth surrogate, area).
5. Convert pixel measurements into real-world units using a reference-object calibration.
6. Predict live body weight using a trained regression model.
7. Report the estimated weight alongside an uncertainty/error range, not a single bare number.
8. Reject or flag images that are unsuitable for reliable estimation, rather than guessing anyway.
9. Generalize to unseen animals, not just animals present in the training data.

---

## 3. Research Questions

- **RQ1:** Can computer vision automatically extract cattle body measurements from a single smartphone image?
- **RQ2:** How close are CV-derived measurements to manual (tape) measurements?
- **RQ3:** Can automatically extracted body measurements predict body weight accurately?
- **RQ4:** Which regression model performs best on this dataset — XGBoost, Random Forest, or a traditional formula baseline?
- **RQ5:** How much does image calibration (reference object) affect measurement accuracy?
- **RQ6:** Which visual/body features contribute most to weight prediction?

**Primary Research Question:**
Can a deep learning-based computer vision pipeline, using only a single smartphone image, estimate the body weight of dairy cattle under real Indian smallholder farm conditions with an accuracy comparable to manual measurement — without specialized hardware or physical contact with the animal?

**Hypothesis:**
A deep learning-based segmentation and regression pipeline, applied to a single smartphone image, can estimate dairy cattle body weight with a correlation of 80–90% (Pearson r) to actual weighing-scale measurements — even when trained on a small pilot dataset of 50–100 animals.

---

## 4. Scope

### In Scope (Current Phase)
- Single smartphone image, side-on view, per animal
- Dairy cows specifically (buffaloes are future work)
- Pilot dataset: 15 animals collected (BAIF field visit, 27 Aug 2026, target was 20–25), scaling to 50–100
- Reference-object calibration for pixel-to-cm conversion
- Explainable pipeline: detection → segmentation → measurement → regression (not end-to-end black-box CNN as primary)
- XGBoost as the primary regression model, benchmarked against Random Forest and a traditional formula baseline
- Web-based portal for field workers: image upload only, no manual measurement in routine use

### Out of Scope (Deferred to Future Work — see Section 14)
- Four-view (front/rear/left/right) multi-image capture
- Video-based capture and automatic best-frame selection
- Depth-camera or 3D reconstruction hardware
- Body Condition Score (BCS) estimation (separate module in the parent project)
- Mobile application (beyond the web-based portal)
- Multi-farm deployment at scale

---

## 5. Literature Review Summary

20 research papers (2018–2026) were reviewed on computer vision-based livestock weight estimation:

- **2018–2020:** 3D cameras and manually extracted measurements fed into regression models. Typical error: ~5% of body weight.
- **2020–2022:** Deep learning (CNN/RNN) began replacing manual feature engineering. RGB-D methods pushed accuracy above 90% in some studies.
- **2023:** Strongest benchmark — 914 real dairy cows using 3D contour features with CatBoost, Random Forest, and SVR, achieving r = 0.94, RMSE 33 kg, MAPE 4.0%.
- **2022 systematic review** of 151 papers found nearly all successful studies rely on the same core features: body length, hip height, hip width, chest girth, and body area.
- **2024–2026:** Recent work shifts toward comparing data modalities (RGB vs. depth vs. RGB-D fusion) and transfer learning from large farms to small, data-limited farms.

---

## 6. Research Gap

- Almost no studies were conducted on Indian smallholder dairy farms.
- Most methods rely on 3D or depth cameras, not a plain smartphone.
- Datasets are large and collected under controlled research-barn conditions.
- No existing work delivers a simple, field-worker-friendly system built for actual deployment.

**This project's contribution:** a smartphone-only, non-invasive, explainable weight estimation system, built and validated specifically for Indian smallholder conditions, delivered through a portal any field worker can use.

---

## 7. Dataset Strategy

A three-stage approach resolves the practical challenge of needing labeled data before BAIF's full dataset is available.

### Stage 1 — Pipeline Bootstrapping (Public Datasets)
Used to build and test the pipeline before field data exists. **Side-view datasets only** are used for this — an earlier candidate (Cows2021, University of Bristol) was found to be captured from an **overhead/top-down camera angle** and has been dropped, since it cannot support side-view length/height/girth feature extraction.

- **Cattle Side and Back View Dataset (Mendeley)** — 72 cattle, side and back view images, annotated with oblique body length, withers height, heart girth, hip length, and body weight. Closest structural match to this project's feature set. (Note: Horqin cattle, a Chinese dual-purpose breed, not Holstein dairy — used for pipeline development only, not final reported results.)
- **Cattle Weight Detection Dataset (Kaggle, "12k")** — side-view images with weight encoded in filenames and a calibration sticker of known size in-frame. Referenced primarily for its **calibration methodology** (known-size reference object in-shot), which mirrors this project's reference-marker approach.

### Stage 2 — Pilot Data Collection (BAIF Field Visit) — COMPLETED
20–25 animals targeted; **15 animals collected** at Urulikanchan farms (27 Aug 2026). BAIF-CRS team performed physical measurement (heart girth, body length, height at withers, stature height) and recorded tape-estimated weight; student captured calibrated images across four angles (left, right, front, back — see note below) and logged data.

**Ground-truth clarification — Tape-Estimated Weight, not a direct scale reading:**
The `Tape_Weight_Kg` value collected is derived from a calibrated livestock weigh tape (heart-girth-based, pre-computed weight values printed on the tape), not a direct weighbridge/scale reading. This is an established field method, but it carries its own typical error margin (~5–10% in published literature) and should be described precisely as **"tape-estimated weight"** throughout this project's documentation and reporting — never as "actual weight" or unqualified "ground truth" — to avoid overclaiming measurement precision, consistent with this project's "girth surrogate" terminology discipline (Section 9).

BAIF has indicated that machine (scale) weight and tape weight were found to be closely aligned during field cross-checks. This is a positive, reportable validation point, but requires a specific figure before it can be cited: **how many animals were cross-checked, the exact discrepancy (kg or %), and which machine/scale was used** are open questions to confirm with BAIF. Once available, this will be reported as:

> *"Tape-estimated weight was cross-validated against [machine/scale] readings for [N] animals during the BAIF field visit, showing a mean discrepancy of approximately [X kg / X%], consistent with published accuracy ranges for calibrated girth-tape methods (typically 5–10%). This confirms tape-estimated weight as an acceptable proxy for ground truth in this pilot phase, pending full scale validation on the larger BAIF dataset."*

**Bonus over original scope — four-view capture already collected:** the pilot data includes left-lateral, right-lateral, front, and back photos per animal (plus a video for 2 of 15 animals), exceeding the single side-view protocol originally scoped. This unlocks the side-view-only vs. multi-view comparison experiment listed as future work (Section 14) as an experiment runnable on this very pilot batch, rather than a future-only item. Breed across this pilot batch is uniform (HF X — crossbred Holstein-Friesian), so this batch alone cannot yet support breed-generalization claims.

### Stage 3 — BAIF Full Dataset
50–100 animals, following the same protocol, provided after BAIF reviews the v1 prototype built on Stage 2 data.

### Ground-Truth Principle
Manual/tape-estimated weight is required **only** to train and validate the model. It is never required during routine field use once the model is deployed — that is the core value proposition of the portal. (See the Ground-Truth Clarification above regarding the precision of tape-estimated weight vs. a direct scale reading.)

---

## 8. Image Capture Protocol (Current Phase — Single Image)

1. **Side-on view** — animal standing squarely, not at an angle.
2. **Full body in frame** — nose to tail, all four legs visible where possible.
3. **Reference marker in-shot** — a marked card/pole of known size, placed near the animal, at roughly the same distance from the camera as the animal, for pixel-to-cm calibration.
4. **Neutral posture preferred, not required** — capture requirements stay forgiving rather than strict, since the animal cannot be expected to hold a studio pose. A natural pause (feeding, at a trough) is sufficient.
5. **Reasonably clean background, daylight lighting.**
6. **Consistent camera distance** — roughly 2–3 meters.
7. **Retake if:** blurry, animal cut off, or strongly angled.

*(Note: video-based capture and multi-frame quality filtering are planned future work — Section 14 — not part of this phase.)*

---

## 9. Computer Vision Pipeline

### Stage 1 — Image Quality Assessment
Before any prediction, check: is an animal present, is it fully visible, is it sufficiently large in frame, is the view approximately side-on, is it too dark/blurry, are multiple animals present? If quality is poor, the system should say so explicitly (e.g., *"Unable to estimate reliably — please upload a clearer side-view image with the full animal visible"*) rather than silently producing an unreliable number.

### Stage 2 — Animal Detection
YOLOv8 (or a segmentation-capable YOLO variant) locates the animal, producing a bounding box and confidence score.

### Stage 3 — Segmentation
YOLOv8-seg produces the actual animal silhouette (not just a bounding box), from which pixel area, width, height, and contour can be calculated.

### Stage 4 — Measurement Extraction
- **Body Length** — distance between shoulder and pin-bone landmarks (or bounding-box proxy in the first prototype iteration).
- **Height at Withers** — vertical distance from withers landmark to ground.
- **Girth Surrogate** — *not* true circumference (a single side image cannot observe the full circumference of the animal). Derived instead from torso height/width/silhouette thickness. This terminology distinction is used consistently in all documentation and reporting to avoid overclaiming measurement precision.
- **Silhouette Area** — total pixel area of the segmented mask, converted to cm² via calibration.

### Stage 5 — Calibration
```
Scale (cm/pixel) = Known reference size (cm) / Reference size in image (pixels)
Length_cm = Length_px × Scale
Area_cm² = Area_px × Scale²   (scale is squared, since area = length × width)
```

### Stage 6 — Weight Prediction
Extracted, calibrated features are passed into the trained regression model (Section 10).

---

## 10. Model Training Strategy

### Experiment Structure
Following a staged-baseline approach, so the project can report *how much* accuracy comes from CV measurement quality versus regression model choice:

| Experiment | Input | Model | Purpose |
|---|---|---|---|
| 1 | Manual measurements | Traditional formula (Schaeffer's) | Non-ML baseline |
| 2 | Manual measurements | Random Forest | Upper-bound baseline: best possible accuracy if measurements were perfect |
| 3 | Manual measurements | XGBoost | Upper-bound baseline, primary model family |
| 4 | CV-extracted measurements | XGBoost | **Primary system** — real-world pipeline performance |
| 5 | CV-extracted measurements | Random Forest | Comparison against primary model |
| 6 | Raw image | CNN/EfficientNet (end-to-end) | Secondary comparison — posture-invariant deep features, not the primary model |

Experiments 2–3 establish the **ceiling**: if even perfect manual measurements can't predict weight well, no amount of CV improvement will fix that. Experiment 4 is the actual deliverable. Experiment 6 tests whether skipping explicit measurement helps or hurts, given the small dataset.

### Primary Model: XGBoost
XGBoost (gradient boosting) is used as the primary regression model for its sequential error-correction, built-in L1/L2 regularization, and strong track record in the literature reviewed. Given the small dataset, tree depth and estimator count are kept conservative, with k-fold cross-validation used to confirm stability across splits rather than trusting a single train/test split.

---

## 11. Data Split Strategy — Preventing Leakage

**Critical rule:** if multiple images exist per animal, the split must be done by **animal ID**, never by individual image.

```
INCORRECT (causes data leakage):
  Training:  Cow A - Image 1, Cow A - Image 2
  Testing:   Cow A - Image 3

CORRECT:
  Training:   Animals 1-16
  Validation: Animals 17-20
  Testing:    Animals 21-25
```

The same animal ID must never appear across more than one split. This is essential for honestly measuring generalization to unseen animals, and is a common oversight in small-dataset livestock CV work — several papers in the literature review did not explicitly confirm this practice.

*(Note: given the pilot's small scale — 15 animals collected — a strict three-way split may leave very small validation/test sets. K-fold cross-validation at the animal-ID level, rather than a fixed three-way split, is the more statistically sound choice until the dataset grows via Stage 3.)*

---

## 12. Evaluation Metrics & Reporting Standard

**Do not report a single "accuracy" percentage.** Weight is a continuous variable — report regression metrics:

- **MAE** (Mean Absolute Error, kg)
- **RMSE** (Root Mean Squared Error, kg) — penalizes larger errors more heavily
- **MAPE** (Mean Absolute Percentage Error, %)
- **R²** — reported alongside MAE/RMSE/MAPE, never alone
- **Pearson correlation (r)** — primary metric for the 80–90% target agreed with BAIF
- **Percentage of predictions within ±5% / ±10% / ±15% of tape-estimated weight** — the most farmer-interpretable metric

**The exact achievable performance must be determined experimentally and not assumed in advance.**

### Confidence Reporting
The portal should never show a single bare number. Preferred output format:
```
Estimated Weight: 482 kg
Expected Range:   455–509 kg
```
The uncertainty range should be calibrated from validation data, not an arbitrary or fabricated confidence percentage.

---

## 13. Handling Unsuitable Images

The system should detect and clearly message the field worker for:
- **Animal too far** (occupies too few pixels) → "Move closer."
- **Animal too close** (not fully visible) → "Move farther away."
- **Multiple animals in frame** → "Please upload an image containing one animal."
- **Occlusion** → "Important body regions are blocked. Please capture a clear side view."
- **Poor lighting / blur** → "Image quality is insufficient."
- **Non-side view** → "Please provide a clearer side-view image."

A system that confidently predicts weight from an unsuitable image is worse than one that asks for a retake.

---

## 14. Future Scope (Explicitly Deferred)

- **Four-view multi-image capture** (front, rear, left, right) — reviewed as a strong methodology for capturing width/circumference information a single side view cannot observe, but deferred since it roughly quadruples per-animal field capture effort beyond the protocol already agreed with BAIF. Recommended as a **v2 experiment**: compare side-view-only vs. left+right vs. all-four-views performance once single-view results are established.
- **Video-based capture** with automatic best-frame selection, reducing reliance on a single well-posed photo.
- **True girth measurement** via multi-view or depth data (current single side-view approach explicitly uses "girth surrogate," not true circumference).
- **Depth-camera / 3D reconstruction** methods (RGB-D, point clouds) for improved accuracy, if hardware becomes available.
- Extension to buffaloes.
- Larger dataset from BAIF, multiple farms, breed diversity.
- Dedicated mobile application (beyond the current web-based portal).
- Integration with the parallel Body Condition Score (BCS) module.

---

## 15. System Architecture

```
Smartphone Image
      ↓
Image Preprocessing
      ↓
Animal Detection (YOLOv8)
      ↓
Segmentation (YOLOv8-seg)
      ↓
Calibration (reference marker)
      ↓
Feature Extraction (Length • Height • Girth Surrogate • Area)
      ↓
Weight Prediction (XGBoost)
      ↓
Estimated Weight + Range
```

### Portal Architecture
- **Frontend:** mobile-friendly web app, phone browser upload, no app-store install needed
- **Backend/API:** Flask or Django REST API running the inference pipeline
- **Database:** animal metadata, images, measurements, predictions, model version (for traceability across retraining)
- **Model Versioning:** every prediction logged against a model version (e.g., `weight_model_v1`), so future retrains can be compared

---

## 16. Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python |
| Image processing | OpenCV |
| Detection/Segmentation | YOLOv8 / YOLOv8-seg (Ultralytics), PyTorch |
| Regression modeling | XGBoost (primary), Scikit-learn (Random Forest comparison) |
| Data processing | NumPy, Pandas |
| Annotation | CVAT / Roboflow |
| Backend | Flask or Django |
| Frontend | Mobile-friendly web app |
| Compute | Laptop or free-tier cloud GPU |

---

## 17. Important Limitations (to state explicitly in the final report)

- Breed, age, sex, and pregnancy-status differences affect body shape independent of frame size
- **Lactation stage/parity affects body weight independent of skeletal size** (see Data Collection Methodology document) — recorded during field visits to explain outliers
- Camera angle, distance, lighting, and occlusion affect measurement accuracy
- Pose variation (see earlier discussion on non-static animals)
- Small dataset size (50–100 animals) limits generalization claims
- Farm-to-farm variation — a model trained on one farm/region may not generalize perfectly elsewhere without broader data

---

## 18. What Not To Do

- Do not train on a handful of animals and claim a single "90% accuracy" figure.
- Do not split individual images across train/test if they belong to the same animal (Section 11).
- Do not assume a fixed pixels-to-cm ratio without per-image calibration.
- Do not claim true girth/circumference from a single side-view image — use "girth surrogate."
- Do not report R² alone, without MAE/RMSE/MAPE alongside it.
- Do not use actual test-set weights during feature engineering or model tuning.
- Do not build only an end-to-end black-box model without a measurement-based, explainable baseline to compare against.

---

## 19. Timeline (4–5 Months)

| Phase | Timing | Focus |
|---|---|---|
| Phase 1 | Month 1 | Literature review, protocol design, pipeline bootstrapping on public datasets |
| Phase 2 | Month 1–2 | BAIF pilot field visit — **completed, 15 animals collected** — data logging, image calibration setup |
| Phase 3 | Month 2–3 | CV pipeline: detection, segmentation, measurement extraction, validation against manual measurements |
| Phase 4 | Month 3 | Model training: baseline experiments (1–3), primary XGBoost pipeline (Experiment 4), comparisons (5–6) |
| Phase 5 | Month 3–4 | BAIF full dataset (50–100 animals) integration, retraining, correlation benchmarking against 80–90% target |
| Phase 6 | Month 4 | Portal build: upload frontend + inference API |
| Phase 7 | Month 4–5 | BAIF review, documentation, methodology write-up, research paper drafting |

---

## 20. Expected Deliverables

1. A validated body weight estimation model with reported correlation, MAE, RMSE, MAPE, and R² against real scale weights — using an animal-ID-based (leakage-free) validation split.
2. A working portal where field workers upload a single image and receive a weight estimate with an uncertainty range — no manual measurement, no extra hardware.
3. A documented experimental comparison (traditional formula vs. Random Forest vs. XGBoost vs. CV-measurement-based vs. end-to-end CNN), showing where prediction accuracy actually comes from.
4. Full methodology documentation, explicitly scoped and limitation-aware, ready to extend into the BCS module, four-view capture, and a future research publication.
