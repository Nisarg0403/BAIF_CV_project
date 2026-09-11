# AI-Based Cattle Live Weight Estimation from Multi-View Images

## Complete Project Plan, AI Specification, CV Pipeline, Dataset Protocol, and Development Roadmap

**Project type:** Computer Vision + Machine Learning + Multi-View Geometry + Regression

**Primary objective:** Estimate cattle live body weight from images/video without requiring the farmer to physically measure the animal every time.

**Primary input:** Four views of the same cattle — **left side, right side, front, rear**.

**Optional future input:** A short video from which the system automatically selects the four best views.

**Ground truth:** BAIF/field records containing actual tape weight and manual morphometric measurements.

---

# 1. Executive Summary

The proposed system will estimate cattle live weight from visual information captured using a smartphone.

The system should not simply perform:

```text
One image -> one weight
```

Instead, the primary research architecture is:

```text
4 views of the same cattle
        |
        v
Image quality + view validation
        |
        v
Cattle detection
        |
        v
Cattle segmentation
        |
        v
Anatomical keypoint detection
        |
        v
Per-view measurements
        |
        v
Multi-view feature fusion
        |
        v
Weight regression model
        |
        v
Estimated live weight + confidence/range
```

The central research hypothesis is:

> Combining complementary views of the same cattle should provide more reliable body-shape information than relying on a single side photograph, especially for width, depth, and body-volume-related features.

---

# 2. Current Problem Observed in the Prototype

The current Streamlit prototype produces values such as:

```text
BAIF ground truth:
Weight       = 578 kg
Chest Girth  = 192 cm
Body Length  = 157 cm
Wither       = 144 cm
Stature      = 143 cm
```

while the image pipeline produced approximately:

```text
Left prediction:
Weight       = 650.7 kg
Girth        = 199.1 cm
Length       = 153.4 cm
Wither       = 137.3 cm
Stature      = 140.5 cm

Right prediction:
Weight       = 611.8 kg
Girth        = 196.4 cm
Length       = 145.1 cm
Wither       = 135.8 cm
Stature      = 139.0 cm

Current average:
Weight       = 631.2 kg
```

Difference from actual:

```text
631.2 - 578 = +53.2 kg

Relative error:
53.2 / 578 * 100 ~= 9.2%
```

This is not a reason to immediately change the final formula. The first task is to determine **where the error originates**.

---

# 3. Critical Diagnostic Experiment — Do This First

Before changing the CV pipeline or retraining anything, perform two controlled tests.

## Test A — Manual Measurements -> Existing Weight Model

Take the actual BAIF measurements:

```text
Chest Girth = 192 cm
Body Length = 157 cm
Wither Height = 144 cm
Stature Height = 143 cm
```

Feed these exact values into the existing weight model, bypassing computer vision.

```text
BAIF manual measurements
          |
          v
Existing preprocessing
          |
          v
Existing weight model
          |
          v
Predicted weight
```

### Interpretation

If the result is close to 578 kg:

> The weight model is broadly valid. The major problem is in image measurement/calibration.

If the result is close to 630 kg or another high value:

> The weight model itself is biased/mismatched, or its feature order/preprocessing is incorrect.

This is the most important first diagnostic.

---

# 4. Test B — Compare Manual vs CV Measurements

Use the same cattle and compare:

```text
Ground truth -> CV measurement
```

For each feature calculate:

- MAE
- RMSE
- MAPE
- Bias / mean signed error
- R² where appropriate

Example:

| Feature | Manual | CV | Absolute Error | Percentage Error |
|---|---:|---:|---:|---:|
| Girth | 192.0 | 197.8 | 5.8 | 3.0% |
| Length | 157.0 | 149.3 | 7.7 | 4.9% |
| Wither | 144.0 | 136.6 | 7.4 | 5.1% |
| Stature | 143.0 | 139.8 | 3.2 | 2.2% |

The measurement model and weight model must then be evaluated separately.

---

# 5. Important Correction: Do Not Treat 2D Side Geometry as True Girth

A side image does not directly measure the circumference around the cattle.

Therefore the feature should initially be named:

```text
Estimated_Chest_Girth_cm
```

or preferably:

```text
Chest_Girth_Surrogate_cm
```

rather than claiming that the value is a direct tape measurement.

The system can estimate circumference-related information using complementary geometry from side and front/rear views.

---

# 6. Four-View Input Design

The primary image set is:

```text
Cattle ID: C001

front.jpg
rear.jpg
left.jpg
right.jpg
```

These are **four views of one animal**, not four independent training samples.

## Required Views

### Left-side profile

Useful for:

- Body length
- Wither height
- Stature height
- Body depth
- Silhouette area
- Torso shape

### Right-side profile

Useful for:

- Body length
- Wither height
- Stature height
- Body depth
- Silhouette area
- Redundancy against side-specific errors

### Front view

Useful for:

- Chest width
- Shoulder width
- Front-body width
- Front silhouette area
- Chest cross-sectional shape

### Rear view

Useful for:

- Rump width
- Hip width
- Rear-body width
- Rear silhouette area
- Rear cross-sectional shape

## Optional 3/4 Views

Existing photographs that are front-left, front-right, or other oblique views should not be discarded.

Label them separately as:

```text
front_left_3q
front_right_3q
rear_left_3q
rear_right_3q
```

Do not force a 3/4 image into the `front` or `side` class if the camera geometry is materially different.

---

# 7. Photo Capture Protocol

The final system will only be as reliable as the image acquisition protocol.

## Preferred cattle posture

The animal should:

- Stand naturally
- Stand approximately upright
- Have all relevant body regions visible
- Avoid extreme head bending
- Avoid lying down
- Avoid walking during capture
- Have minimum obstruction

For side views:

```text
Camera
  |
  |  approximately perpendicular
  v
----------------------
        CATTLE
----------------------
```

For front/rear views:

The camera should face the chest/rump approximately straight-on.

## Camera consistency

Where possible, keep approximately consistent:

- Camera height
- Camera distance
- Focal length / zoom
- Image resolution
- Lighting
- Background conditions

## Important

Do not require a perfect studio environment for the final research. The model should eventually be robust to real farm environments, but the first dataset should be controlled enough to learn reliable geometry.

---

# 8. Physical Calibration Strategy

The formula

```text
measurement_cm = measurement_px * Scale
```

is valid only if `Scale` is known for the particular image.

A fixed global scale is not reliable for arbitrary smartphone photographs because pixel size changes with camera distance and perspective.

## Recommended first solution: known reference marker

Place a known-size reference in the capture environment.

For example:

```text
Known reference length = 100 cm
Detected pixel length   = 500 px

Scale = 100 / 500
      = 0.20 cm/px
```

Then:

```text
Body Length (cm) = Body Length (px) * Scale
Body Depth (cm)  = Body Depth (px) * Scale
Body Width (cm)  = Body Width (px) * Scale
```

For area:

```text
Area_cm2 = Area_px * Scale^2
```

## Future markerless calibration

After the controlled version works, investigate:

- Camera calibration
- Known camera distance
- Perspective correction
- Homography
- Multi-view geometry
- Monocular depth estimation
- Stereo/multi-view reconstruction
- Learned scale estimation

Do not make markerless calibration the first milestone if it prevents reliable results.

---

# 9. Computer Vision Pipeline

## Stage 1 — Input Validation

For four photos:

```text
Front image
Rear image
Left image
Right image
```

Check:

- File validity
- Resolution
- Blur
- Exposure
- Cattle visibility
- Full-body visibility
- Correct view orientation
- Excessive occlusion

Output:

```text
View quality score
View classification
Warnings
```

---

# 10. Stage 2 — Cattle Detection

Use an object detector to locate the cattle.

Possible models:

- YOLO-family detector
- RT-DETR
- Other lightweight detector

Output:

```text
Bounding box
Confidence
```

The bounding box should only be used for localization, not as the final anatomical measurement.

---

# 11. Stage 3 — Cattle Segmentation

Use instance/semantic segmentation to isolate the animal from the background.

The current green-mask output is a good prototype, but the segmentation boundary must be improved.

The mask should exclude:

- Ground shadows
- Feed troughs
- Ropes where possible
- Walls
- Other cattle
- Background objects

The mask should include:

- Head
- Neck
- Torso
- Legs
- Tail when appropriate

The system should retain both:

```text
Full animal mask
```

and a cleaned body/torso mask where necessary for morphometry.

---

# 12. Stage 4 — Anatomical Keypoint Detection

This is one of the most important upgrades to the current prototype.

Do not rely only on:

```text
x_max - x_min
```

or

```text
highest silhouette pixel
```

to define anatomical measurements.

Instead annotate and detect meaningful anatomical landmarks.

## Recommended side-view keypoints

At minimum investigate:

```text
1. Withers
2. Shoulder reference point
3. Elbow / chest reference
4. Brisket reference
5. Hip / hook reference
6. Pin bone reference
7. Ground contact / hoof reference
8. Tail base
9. Head reference
```

Exact anatomical definitions must be finalized with a cattle/veterinary/animal-science expert and then applied consistently.

---

# 13. Body Length Definition

Do not define body length simply as the entire silhouette width.

A robust approach is to define consistent anatomical endpoints.

Conceptually:

```text
Shoulder reference ●----------------------● Pin/hip reference
                  <------ length ------->
```

Then:

```text
BodyLength_px = distance(shoulder_keypoint, rear_keypoint)
```

After calibration:

```text
BodyLength_cm = BodyLength_px * Scale
```

If left and right side estimates are both reliable:

```text
BodyLength_final = median(left, right)
```

Use reliability weighting if one side has poorer visibility.

---

# 14. Wither Height Definition

The withers are the highest anatomical region of the shoulder, not simply the highest pixel anywhere on the back.

Use:

```text
Withers keypoint
       ●
       |
       |
       |
       ● ground
```

Then:

```text
WitherHeight_px = y_ground - y_withers
```

and:

```text
WitherHeight_cm = WitherHeight_px * Scale
```

The exact withers point should be annotated consistently by trained annotators.

---

# 15. Stature Height Definition

Define the anatomical reference point first.

Then:

```text
StatureHeight_px = y_ground - y_stature_reference
```

and:

```text
StatureHeight_cm = StatureHeight_px * Scale
```

Do not allow the system to choose an arbitrary highest silhouette point.

---

# 16. Body Depth

From a side profile, estimate the vertical depth of the torso at a standardized anatomical cross-section.

For example:

```text
Top of torso
     ●
     |
     |  Body Depth
     |
     ● lower torso/brisket reference
```

The cross-section position must be standardized across cattle.

Possible locations should be selected with animal-science guidance, especially if the intended measurement is related to heart girth/chest girth.

---

# 17. Front Width

From the front view, detect the left and right boundaries of the chest at the standardized cross-section.

```text
      ●-----------●
          width
```

Then:

```text
ChestWidth_cm = ChestWidth_px * Scale
```

This provides information unavailable from a single side photograph.

---

# 18. Rear Width

From the rear view, estimate:

- Rump width
- Hip width
- Rear-body width

These can be included as separate features rather than forcing one width value.

---

# 19. Girth / Circumference Surrogate

Do not use the following:

```text
Blue vertical measurement * 2 + red vertical measurement
```

as a physical circumference.

There is no valid geometric reason to multiply a leg/height measurement by two to obtain cattle chest circumference.

Instead use complementary body geometry.

## Elliptical approximation

If:

```text
D = body depth
W = body width
```

and the cross-section is approximated by an ellipse:

```text
a = D / 2
b = W / 2
```

Then a Ramanujan approximation is:

```text
C ~= pi * [3(a+b) - sqrt((3a+b)(a+3b))]
```

This can be used as a **girth surrogate**, not automatically assumed to equal tape girth.

The model should ultimately learn whether this feature improves weight estimation.

---

# 20. Multi-View Feature Fusion

Do not calculate four independent final weights and simply average them as the primary architecture.

Current approach:

```text
Left image  -> 650.7 kg
Right image -> 611.8 kg

Average -> 631.2 kg
```

Recommended approach:

```text
Left image
    |
    +--> Left features -----+
                            |
Right image                 |
    |                       |
    +--> Right features ----+
                            |
Front image                 +--> Feature fusion --> Weight model
    |                       |
    +--> Front features ----+
                            |
Rear image                  |
    |                       |
    +--> Rear features -----+
```

This allows the model to learn how different measurements interact.

---

# 21. Recommended Feature Vector

## Left side

```text
body_length_left_cm
body_depth_left_cm
wither_height_left_cm
stature_height_left_cm
silhouette_area_left_cm2
```

## Right side

```text
body_length_right_cm
body_depth_right_cm
wither_height_right_cm
stature_height_right_cm
silhouette_area_right_cm2
```

## Front

```text
chest_width_cm
shoulder_width_cm
front_body_depth_cm
front_silhouette_area_cm2
```

## Rear

```text
rump_width_cm
hip_width_cm
rear_silhouette_area_cm2
```

## Derived features

```text
mean_body_length_cm
mean_wither_height_cm
mean_stature_height_cm

left_right_length_difference
left_right_height_difference
left_right_area_difference

front_rear_width_ratio
width_to_length_ratio
depth_to_length_ratio
height_to_length_ratio

estimated_girth_surrogate_cm
```

Only retain features that are available consistently and demonstrably improve validation performance.

---

# 22. Weight Prediction Model

The first main model should be an interpretable tabular regression model using the fused measurements.

Recommended candidates:

1. XGBoost Regressor
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Extra Trees Regressor
5. Linear/Polynomial regression as a baseline

## Recommended primary model

```text
Multi-view measurements
          |
          v
Feature preprocessing
          |
          v
XGBoost Regressor
          |
          v
Weight (kg)
```

XGBoost is recommended because it can model nonlinear relationships between body dimensions and weight.

---

# 23. Important XGBoost Clarification

Do not assume that XGBoost automatically uses:

```text
Weight proportional to Girth^2 * Length
```

That relationship applies to traditional cattle-weight formulas, not automatically to XGBoost.

XGBoost learns relationships from the training data.

Therefore the actual model package must be inspected for:

- Model type
- Feature names
- Feature order
- Scaling
- Normalization
- Feature engineering
- Girth squared feature
- Log transformations
- Calibration/correction layers
- Ensemble structure

Do not manually alter the model until this audit is complete.

---

# 24. Traditional Formula Baseline

A traditional morphometric formula should be included as a research baseline.

For example, if an appropriate cattle formula is selected from the literature/animal-science guidance:

```text
Weight_baseline = f(girth, length)
```

The project should compare:

```text
Traditional formula
        VS
Manual measurements + ML
        VS
CV measurements + ML
        VS
Four-view CV + ML
```

The exact traditional formula must be documented with its source and validated for the cattle population being studied.

Do not present a formula from another breed/population as universally valid.

---

# 25. Dataset Design

The dataset should be cattle-centric.

Recommended structure:

```text
dataset/
|
+-- C001/
|   +-- front.jpg
|   +-- rear.jpg
|   +-- left.jpg
|   +-- right.jpg
|
+-- C002/
|   +-- front.jpg
|   +-- rear.jpg
|   +-- left.jpg
|   +-- right.jpg
|
+-- C003/
    +-- ...
```

Metadata example:

```csv
cattle_id,front_image,rear_image,left_image,right_image,actual_weight_kg,body_length_cm,chest_girth_cm,wither_height_cm,stature_height_cm
C001,front.jpg,rear.jpg,left.jpg,right.jpg,578,157,192,144,143
```

Additional useful metadata:

```text
breed
sex
age
BCS
lactation status where appropriate
capture date
session ID
farm/site ID
camera/device
view quality
occlusion score
```

---

# 26. Your Existing BAIF Data Is Extremely Valuable

For every cattle, aim to have:

```text
ONE CATTLE ID
       |
       +-- Front image
       +-- Rear image
       +-- Left image
       +-- Right image
       |
       +-- Manual body length
       +-- Manual chest girth
       +-- Manual wither height
       +-- Manual stature height
       |
       +-- Actual/tape weight
```

This is a supervised learning sample.

Do not split the four images of one cattle across train and test sets.

---

# 27. Prevent Data Leakage

This is critical.

If C001 has four images, all four must belong to exactly one dataset split.

Correct:

```text
Train:
C001, C002, C003 ...

Validation:
C101, C102 ...

Test:
C151, C152 ...
```

Incorrect:

```text
Train:
C001-left
C001-front

Test:
C001-right
C001-rear
```

The incorrect setup can make the model appear much more accurate than it really is.

---

# 28. Dataset Size Strategy

If the current dataset is small, do not immediately build a deep end-to-end model.

Recommended progression:

### Phase 1

Use the existing manually measured dataset to establish a regression baseline.

### Phase 2

Use image-derived measurements + XGBoost.

### Phase 3

Add all four views.

### Phase 4

Train a learned multi-view image model if the dataset is sufficiently large.

A deep neural network should not be the first choice simply because the project is called AI.

Reliable labels and measurement definitions are more important.

---

# 29. Annotation Protocol

Create an annotation specification before labeling hundreds of images.

Each side image should contain consistent keypoints.

Recommended annotation categories:

```text
withers
shoulder
hip/hook
pin/rear reference
brisket reference
hoof/ground reference
body top reference
body bottom reference
```

Front/rear images:

```text
left chest boundary
right chest boundary
shoulder boundaries
left hip/rump boundary
right hip/rump boundary
```

Each annotation should have:

```text
x
 y
visibility
confidence/quality
```

The final anatomical definitions should be approved by an animal-science expert.

---

# 30. Segmentation Quality Requirements

Measure segmentation independently from weight.

Metrics:

- IoU
- Dice coefficient
- Precision
- Recall
- Boundary quality where useful

Create a validation set with difficult examples:

- Black cattle
- White cattle
- Mixed-color cattle
- Shadows
- Muddy floors
- Concrete floors
- Feed troughs
- Ropes
- Other cattle
- Low light
- Strong sunlight

The goal is not merely a visually attractive green overlay. The mask must support accurate morphometry.

---

# 31. Measurement Evaluation

For every physical measurement calculate:

```text
MAE
RMSE
MAPE
Mean Bias
Median Absolute Error
R²
```

Example:

```text
Body Length:
MAE = X cm
MAPE = X%

Wither Height:
MAE = X cm
MAPE = X%

Girth surrogate:
MAE = X cm
MAPE = X%
```

This allows you to identify which CV measurement contributes most to the final weight error.

---

# 32. Weight Evaluation

Do not report only "accuracy".

For regression, report:

```text
MAE (kg)
RMSE (kg)
MAPE (%)
R²
Median Absolute Error
Bias
```

Also report practical tolerance:

```text
Percentage within ±5%
Percentage within ±10%
Percentage within ±15%
```

For example:

```text
MAE = 24.1 kg
MAPE = 4.8%
R² = 0.91
87% within ±10%
```

The numbers above are examples only, not target results.

---

# 33. Do Not Claim "80–90% Accuracy" Without Defining It

The project goal may be expressed operationally as:

> Estimate cattle live weight with a useful error range and aim for approximately 80–90% of predictions to fall within an agreed tolerance band such as ±10% of actual weight.

This is clearer than simply saying:

```text
AI accuracy = 90%
```

For a regression problem, the tolerance definition must be explicit.

---

# 34. Error Decomposition

Every prediction should be traceable through the pipeline.

```text
Final weight error
       |
       +-- Segmentation error
       |
       +-- Keypoint error
       |
       +-- Pixel-to-cm calibration error
       |
       +-- Measurement extraction error
       |
       +-- Feature engineering error
       |
       +-- Weight model error
       |
       +-- Input/posture/view-quality error
```

The research report should quantify these wherever possible.

---

# 35. Current Prototype: Specific Problems to Fix

## Problem 1 — Left/right predictions differ too much

Current example:

```text
Left  = 650.7 kg
Right = 611.8 kg
Difference = 38.9 kg
```

Potential causes:

- Different camera distance
- Different perspective
- Head position
- Segmentation quality
- Body posture
- Anatomical keypoint error
- Calibration error
- Side-specific model bias

Solution:

- Standardize capture
- Use anatomical keypoints
- Normalize camera geometry
- Use both side features in one fused model
- Add left/right consistency features

---

# 36. Current Prototype: Body Length Problem

Current example:

```text
Actual = 157 cm
Left CV = 153.4 cm
Right CV = 145.1 cm
```

The right-side error is particularly large.

Likely issue:

```text
bounding box / silhouette span
```

being used as an anatomical body-length measurement.

Fix:

```text
keypoint shoulder -> rear anatomical reference
```

rather than:

```text
x_max - x_min
```

---

# 37. Current Prototype: Wither Height Problem

Current example:

```text
Actual = 144 cm
CV average = 136.6 cm
```

Do not use the highest pixel of the entire animal silhouette.

Use a dedicated withers keypoint.

Then estimate the ground reference from the appropriate hoof/ground geometry.

---

# 38. Current Prototype: Girth Problem

Current example:

```text
Actual = 192 cm
CV average = 197.8 cm
```

The numerical difference is only about 3%, but the physical interpretation needs improvement.

Use:

```text
Side body depth
+
Front/rear body width
        |
        v
Girth surrogate
```

and validate this surrogate against real tape girth across many cattle.

---

# 39. Model Audit: Existing `baif_multimodel_pack.pkl`

Before retraining, inspect the current model package.

Determine:

```text
Model type
Expected input features
Feature order
Feature names
Scaler
Encoder
Transformations
Feature engineering
Girth squared or not
Model version
Training population
Target definition
```

The code should produce an audit report such as:

```text
Model type: XGBoostRegressor
Features: [...]
Scaler: StandardScaler / None
Target: Weight_kg
Transformations: [...]
```

This audit is required before modifying the prediction logic.

---

# 40. Recommended Software Architecture

```text
project/
|
+-- app/
|   +-- streamlit_app.py
|
+-- src/
|   +-- detection.py
|   +-- segmentation.py
|   +-- view_classifier.py
|   +-- keypoints.py
|   +-- calibration.py
|   +-- morphometry.py
|   +-- multiview.py
|   +-- feature_engineering.py
|   +-- weight_model.py
|   +-- uncertainty.py
|   +-- quality.py
|
+-- models/
|   +-- detector.*
|   +-- segmenter.*
|   +-- keypoint_model.*
|   +-- weight_model.*
|
+-- data/
|   +-- images/
|   +-- annotations/
|   +-- metadata/
|
+-- notebooks/
|   +-- 01_data_audit.ipynb
|   +-- 02_measurement_validation.ipynb
|   +-- 03_weight_baseline.ipynb
|   +-- 04_multiview_model.ipynb
|   +-- 05_error_analysis.ipynb
|
+-- tests/
|
+-- configs/
|
+-- requirements.txt
+-- README.md
```

---

# 41. Streamlit Application Flow

The farmer-facing application should eventually look like:

```text
+---------------------------------------+
|       CATTLE WEIGHT ESTIMATOR         |
+---------------------------------------+
|                                       |
| Upload 4 photos                       |
| [Front] [Rear] [Left] [Right]        |
|                                       |
|            [Analyze]                  |
|                                       |
+---------------------------------------+
```

Then:

```text
Checking images...
       |
       v
Detecting cattle...
       |
       v
Checking posture...
       |
       v
Extracting body measurements...
       |
       v
Combining four views...
       |
       v
Estimating weight...
```

Final output:

```text
Estimated Weight

        590 kg

Expected range
        565–615 kg

Measurements
------------------------
Body Length       155.8 cm
Girth Surrogate   193.5 cm
Wither Height     142.1 cm
Stature Height    141.8 cm

Image quality: GOOD
Confidence: HIGH/MEDIUM/LOW
```

The exact confidence/range must be calibrated from validation data; do not invent a fixed percentage.

---

# 42. Video Workflow

Video should be an extension of the four-view system, not a completely separate prediction system.

```text
Video upload
     |
     v
Frame extraction
     |
     v
View/orientation classification
     |
     +--> Best front frame
     +--> Best rear frame
     +--> Best left frame
     +--> Best right frame
     |
     v
Multi-view CV pipeline
     |
     v
Weight estimate
```

Select frames based on:

- Blur
- Full-body visibility
- View angle
- Occlusion
- Segmentation quality
- Posture

---

# 43. Research Experiments

The project should include controlled comparisons.

## Experiment 1 — Traditional formula

```text
Manual girth + manual length
       -> traditional formula
```

## Experiment 2 — Manual measurements + ML

```text
Manual measurements
       -> XGBoost
```

## Experiment 3 — Single side image + CV measurements

```text
Left image
   -> CV measurements
   -> XGBoost
```

## Experiment 4 — Left + right

```text
Left + right features
       -> XGBoost
```

## Experiment 5 — Four views

```text
Front + rear + left + right
       -> feature fusion
       -> XGBoost
```

## Experiment 6 — Direct multi-view deep model

Only after sufficient data:

```text
Four images
     -> CNN/ViT encoders
     -> feature fusion
     -> regression head
     -> weight
```

The research question is:

> Does multi-view visual information significantly improve weight estimation over single-view estimation?

---

# 44. Ablation Study

Perform feature ablation to discover what actually matters.

Compare:

```text
Length only
Length + Girth
Length + Girth + Height
All side features
Left + Right
Front + Rear
All four views
All four views + derived features
```

This prevents unnecessary features from being added just because they are available.

---

# 45. Cross-Validation

For small datasets, use cattle-level cross-validation.

Example:

```text
5-fold GroupKFold
Group = cattle_id
```

This prevents images from the same cattle appearing in both training and validation folds.

For a final test set, keep completely unseen cattle aside.

---

# 46. Weight Bias Analysis

Calculate:

```text
Error = Predicted - Actual
```

Plot error against:

- Actual weight
- Girth
- Body length
- Breed
- BCS
- Age
- Image quality
- View quality

Check whether the model systematically:

- Overestimates lighter cattle
- Underestimates heavier cattle
- Overestimates certain breeds
- Fails with poor posture
- Fails with black/dark cattle

---

# 47. Calibration of Prediction Range

The UI should eventually report a prediction interval or expected error range.

For example:

```text
Estimated weight: 590 kg
Expected error: approximately ±25 kg
```

The ±25 kg value must be estimated from validation/test residuals.

Possible approaches:

- Residual-based interval
- Quantile regression
- Conformal prediction
- Ensemble uncertainty

Do not label an arbitrary range as confidence.

---

# 48. Handling Poor Inputs

The system should reject or warn about bad inputs rather than produce a confident-looking number.

Examples:

```text
Right-side image too blurry.
Please upload another image.
```

```text
Only part of the cattle is visible.
Please capture the full body.
```

```text
Cattle posture is unsuitable for reliable measurement.
Please wait until the cattle is standing naturally.
```

This can improve real-world reliability more than simply making the regression model larger.

---

# 49. Recommended Confidence Logic

Confidence should be based on measurable signals:

```text
Segmentation quality
+
Keypoint confidence
+
View completeness
+
View consistency
+
Model uncertainty
+
Distance from training distribution
```

Example:

```text
HIGH
All views good
Keypoints confident
Low multi-view disagreement

MEDIUM
One weak view
Moderate keypoint uncertainty

LOW
Multiple poor views
Strong disagreement
Out-of-distribution appearance
```

---

# 50. Immediate Development Roadmap

## Phase 0 — Existing System Audit

Tasks:

- Inspect current repository
- Inspect `baif_multimodel_pack.pkl`
- Identify current model input features
- Identify preprocessing
- Identify current girth calculation
- Identify current scale calculation
- Record exact current prediction pipeline

Deliverable:

```text
CURRENT_PIPELINE_AUDIT.md
```

---

## Phase 1 — Ground Truth Data Audit

Tasks:

- Clean BAIF Excel data
- Standardize column names
- Verify units
- Verify cattle IDs
- Match images to cattle IDs
- Identify missing measurements
- Identify duplicate records
- Check weight plausibility
- Check outliers

Deliverable:

```text
clean_cattle_dataset.csv
```

---

## Phase 2 — Baseline Weight Model

Build:

```text
Manual measurements -> weight
```

Models:

- Linear regression
- Random Forest
- XGBoost

Evaluate:

```text
MAE
RMSE
MAPE
R²
±5%
±10%
±15%
```

This becomes the baseline.

---

## Phase 3 — Current CV Measurement Validation

Run the existing image pipeline on labeled cattle.

Create:

```text
Actual measurement
vs
CV measurement
```

Evaluate each measurement independently.

Do not retrain the weight model yet.

---

## Phase 4 — Fix Calibration

Implement a reliable physical scale strategy.

First version:

```text
Known reference marker
```

Then validate:

```text
pixel measurement -> cm
```

against manual measurements.

---

## Phase 5 — Fix Anatomical Measurements

Replace bounding-box-only measurements with keypoint-based measurements.

Priority order:

```text
1. Body Length
2. Wither Height
3. Stature Height
4. Body Depth
5. Front Width
6. Rear Width
7. Girth surrogate
```

---

## Phase 6 — Build Four-View Dataset

For each cattle:

```text
front
rear
left
right
```

plus:

```text
actual weight
manual measurements
metadata
```

Ensure cattle-level dataset splitting.

---

## Phase 7 — Multi-View Feature Fusion

Create one feature vector per cattle.

Example:

```text
C001
|
+-- Left features
+-- Right features
+-- Front features
+-- Rear features
+-- Derived features
|
v
XGBoost
|
v
Weight
```

---

## Phase 8 — Retrain and Compare

Compare:

```text
Manual + XGBoost
CV side + XGBoost
Left + Right + XGBoost
Four-view + XGBoost
```

Select the best model based on unseen cattle performance.

---

## Phase 9 — Error Correction

Only after the validation results are available:

- Analyze systematic bias
- Add justified features
- Tune hyperparameters
- Improve calibration
- Improve segmentation
- Improve keypoints
- Consider residual correction

Do not simply add a constant correction such as:

```text
predicted_weight - 53.2
```

based on one cattle.

---

## Phase 10 — Video Support

Implement:

```text
video -> frame selection -> four canonical views -> same model
```

---

## Phase 11 — Farmer-Facing UI

Move from research/debug UI toward:

```text
Upload
  -> Analyze
  -> Measurements
  -> Estimated weight
  -> Expected range
  -> Quality warning
```

Keep technical diagnostic information available in an expandable section for research/development.

---

# 51. Recommended First Model Architecture

For the current dataset, the most practical architecture is:

```text
                 FRONT IMAGE
                      |
               Detection/Mask
                      |
               Front features
                      |
                      +-------+
                              |
                 REAR IMAGE   |
                      |       |
               Detection/Mask |
                      |       |
                Rear features +
                              |
                LEFT IMAGE    |
                      |       |
               Detection/Mask |
                      |       |
                Left features +----> Feature Fusion
                              |
               RIGHT IMAGE    |
                      |       |
               Detection/Mask |
                      |       |
               Right features +
                              |
                              v
                       Derived Features
                              |
                              v
                           XGBoost
                              |
                              v
                         Weight (kg)
```

This is the recommended first serious version because it is:

- Explainable
- Easier to debug
- Suitable for tabular datasets
- Less data-hungry than an end-to-end deep model
- Compatible with your existing measurement records

---

# 52. Advanced Future Architecture

After the feature-based system is strong, investigate:

```text
Front image -> Encoder --+
Rear image  -> Encoder --+
Left image  -> Encoder --+--> Attention/Fusion --> Regression --> Weight
Right image -> Encoder --+
```

Possible encoders:

- CNN
- EfficientNet
- ConvNeXt
- Vision Transformer
- Lightweight mobile vision backbone

This model should be compared against the explainable measurement-based system, not automatically assumed to be better.

---

# 53. What NOT to Do

Do not:

1. Train on images without reliable cattle IDs.
2. Split views of the same cattle across train/test.
3. Use one fixed pixel-to-cm scale for arbitrary photos.
4. Call a 2D side estimate "true girth" without validation.
5. Multiply leg/height measurements by two to obtain circumference.
6. Use bounding-box width as anatomical body length without validation.
7. Use the highest silhouette pixel as withers without anatomical definition.
8. Tune the model to one example cattle.
9. Subtract 53.2 kg from every prediction because of this one example.
10. Report only an "accuracy percentage" for a regression model.
11. Build a deep end-to-end model before establishing reliable baselines.
12. Average independent view predictions as the final multi-view architecture.
13. Hide poor image quality from the farmer while still returning a highly confident prediction.

---

# 54. Final Research Evaluation Table

The final report should contain a table similar to:

| Model | Input | MAE (kg) | RMSE (kg) | MAPE (%) | R² | Within ±10% |
|---|---|---:|---:|---:|---:|---:|
| Traditional formula | Manual measurements | TBD | TBD | TBD | TBD | TBD |
| XGBoost baseline | Manual measurements | TBD | TBD | TBD | TBD | TBD |
| CV single-side | Left/right image | TBD | TBD | TBD | TBD | TBD |
| CV two-side | Left + right | TBD | TBD | TBD | TBD | TBD |
| CV four-view | Front + rear + left + right | TBD | TBD | TBD | TBD | TBD |
| Deep multi-view | Four images | TBD | TBD | TBD | TBD | TBD |

The four-view model should only be called better if it demonstrates better performance on unseen cattle.

---

# 55. Final System Definition

The completed system should be defined as:

> A multi-view computer vision and machine learning system that estimates cattle live body weight from four smartphone photographs by detecting and segmenting the cattle, locating anatomical landmarks, estimating physical body dimensions, fusing complementary measurements across views, and predicting weight using a supervised regression model trained against field-measured cattle weight.

The system should provide:

```text
Input
-----
4 cattle images OR video

CV
--
Detection
Segmentation
View classification
Keypoint detection
Calibration
Morphometry

Measurements
-----------
Body Length
Body Depth
Chest Width
Rump Width
Wither Height
Stature Height
Silhouette Area
Girth Surrogate

ML
--
Multi-view feature fusion
XGBoost / comparison models

Output
------
Estimated weight (kg)
Expected error/range
Measurement summary
Input quality
Confidence level
```

---

# 56. Success Criteria

The project should be considered successful only after demonstrating all of the following on unseen cattle:

## Computer Vision

- Reliable cattle segmentation
- Consistent anatomical keypoints
- Acceptable physical measurement errors
- Robustness across both left and right sides

## Machine Learning

- Better performance than a simple baseline
- No cattle-level data leakage
- Stable cross-validation results
- Acceptable MAE/MAPE/RMSE
- Clear error analysis

## Multi-View Benefit

Demonstrate whether:

```text
Single side
    <
Left + Right
    <
Four views
```

in performance, or explain scientifically if the additional views do not improve the model.

## Farmer Usability

The farmer should ideally be able to:

```text
Capture/upload images
       ->
Wait for analysis
       ->
Receive estimated weight
       ->
See measurement summary and reliability warning
```

without manually entering body measurements.

---

# 57. Immediate Next Actions

Do these in this exact order:

### Action 1

**Audit the current `baif_multimodel_pack.pkl` and prediction code.**

Find the exact model, feature order, preprocessing, and current formula.

### Action 2

Run the **manual-measurement diagnostic**:

```text
192 cm girth
157 cm length
144 cm wither
143 cm stature
```

Feed these into the current model.

### Action 3

Record the resulting predicted weight.

This determines whether the main problem is:

```text
CV measurement
OR
weight model
```

### Action 4

Run the current CV system over a representative set of BAIF cattle and create a measurement-error report.

### Action 5

Fix calibration and anatomical keypoints.

### Action 6

Create the four-view dataset at cattle-ID level.

### Action 7

Build a four-view feature-fusion XGBoost model.

### Action 8

Compare single-view, two-view, and four-view performance.

### Action 9

Only after these experiments, optimize the Streamlit application.

### Action 10

Add video-based automatic view selection as the next product feature.

---

# 58. Core Principle

The goal is **not**:

```text
Make this one cow's prediction equal 578 kg.
```

The goal is:

```text
Build a model that consistently predicts unseen cattle
accurately using reliable visual measurements.
```

The correct research loop is:

```text
Field data
   ↓
Ground-truth validation
   ↓
CV measurement accuracy
   ↓
Weight-model accuracy
   ↓
Multi-view fusion
   ↓
Unseen-cattle testing
   ↓
Error analysis
   ↓
Iteration
```

The **four-view approach should be the primary architecture**, while the existing single-side Streamlit system should be treated as the baseline/prototype.

---

# 59. One-Line Project Pipeline

```text
4 Cattle Photos -> Quality Check -> Detection -> Segmentation -> Anatomical Keypoints -> Calibrated Measurements -> Multi-View Feature Fusion -> XGBoost Regression -> Estimated Live Weight + Reliability Range
```

---

# 60. Final Deliverables

The project should ultimately produce:

```text
1. Clean BAIF cattle dataset
2. Four-view image dataset
3. Annotation/keypoint dataset
4. Segmentation model
5. Keypoint model
6. Calibration module
7. Morphometric measurement module
8. Four-view feature-fusion module
9. Weight regression model
10. Evaluation notebook/report
11. Error-analysis report
12. Streamlit/web application
13. Video-to-four-view module
14. Model card/documentation
15. Research paper/thesis results
```

---

## Final Recommendation

**Do not try to improve the current 631.2 kg result by changing a formula immediately.**

First answer this question experimentally:

```text
Actual BAIF measurements
        ↓
Current weight model
        ↓
What weight does it produce?
```

That one test will tell you whether the major problem is the **weight model** or the **CV measurement pipeline**.

Then improve the system in this order:

```text
1. Model audit
2. Manual-measurement baseline
3. CV measurement evaluation
4. Calibration
5. Anatomical keypoints
6. Four-view dataset
7. Multi-view feature fusion
8. XGBoost retraining
9. Unseen-cattle evaluation
10. Error/uncertainty calibration
11. Farmer-facing UI
12. Video support
```

This gives the project a defensible scientific methodology rather than tuning the system to individual examples.
