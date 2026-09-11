# Computer Vision-Based Body Weight Estimation of Dairy Cattle

## A Deep Learning Approach for Indian Smallholder Dairy Systems

**Organization:** BAIF Development Research Foundation  
**Student:** Nisarg Vakharia  
**Program:** FY MSc. (AI/ML)  
**Faculty Guide:** Dr. Jayashree Prasad  
**Institution:** MIT School of Computing  
**Project Type:** Computer Vision + Machine Learning/Deep Learning + Web Prototype

---

# 1. Project Overview

The project aims to develop a **non-contact, smartphone-based system for estimating the body weight of dairy cattle from images**.

The central idea is:

> **Smartphone image of a dairy cow → cattle segmentation → body/morphometric feature extraction → weight prediction → estimated body weight**

The system is intended primarily for **Indian smallholder dairy farming conditions**, where access to weighing scales or weighbridges may be limited.

Body weight is an important parameter for:

- Feeding decisions
- Breeding-related decisions
- Health and veterinary monitoring
- Monitoring changes in an animal's physical condition
- Supporting field-worker decision making

Traditional alternatives such as weighing scales and manual girth-tape measurements can be expensive, unavailable, time-consuming, require trained personnel, or require physical handling of the animal.

The proposed system aims to provide a practical, low-cost alternative using an ordinary smartphone.

---

# 2. Core Problem Statement

Smallholder dairy farmers often lack reliable and convenient methods for measuring cattle body weight.

The project therefore asks:

> **Can the body weight of a dairy cow be estimated accurately enough from a normal smartphone image, without physically weighing or manually measuring the animal?**

The project is not intended to replace certified weighing scales for applications requiring legally or scientifically certified weight. It is intended as a **practical estimation and field-support system**.

---

# 3. Main Project Objectives

## Objective 1 — Smartphone-Based Weight Estimation

Develop a computer-vision system capable of estimating dairy cattle body weight from smartphone images.

## Objective 2 — Build a Pilot Dataset

Develop a dataset of approximately **50–100 real dairy cattle**, where every animal has:

- Smartphone images
- Corresponding actual body-weight measurements obtained using a weighing scale

The actual scale measurement is the **ground truth**.

## Objective 3 — Cattle Segmentation

Automatically separate the cattle from the surrounding background using a deep-learning-based segmentation approach.

## Objective 4 — Morphometric Feature Extraction

Extract useful image-derived cattle features such as:

- Body length
- Body height
- Body girth
- Body area

## Objective 5 — Weight Prediction

Train and compare machine-learning/deep-learning regression approaches to learn the relationship between image-derived features and actual body weight.

## Objective 6 — Prototype Development

Develop a simple web-based interface through which a field worker can upload a cattle image and receive an estimated body weight.

## Objective 7 — Real-World Validation

After the initial prototype is working, use BAIF's real-world cattle data to further validate and improve the system.

## Objective 8 — Future Video-Based Extension

After establishing a working image-based system, extend the approach toward **video-based weight estimation**, allowing the system to handle natural animal movement and posture variation.

---

# 4. Overall Development Strategy

The project will be developed in stages rather than attempting to collect the entire BAIF dataset before beginning implementation.

```text
University of Bristol 2021 Dataset
            ↓
Initial Model Development
            ↓
Initial Experiments
            ↓
Smartphone Data Collection
50–100 Real Cattle
            ↓
Model Adaptation / Validation
            ↓
Working Image-Based Prototype
            ↓
Target: Strong Correlation With Actual Weight
            ↓
BAIF Dataset
            ↓
Real-World Validation / Improvement
            ↓
Final Image-Based System
            ↓
Future Video-Based Extension
```

This staged approach allows the computer-vision pipeline to be developed before the BAIF field dataset is available.

---

# 5. Phase 1 — University of Bristol 2021 Dataset

## Purpose

The first stage will use the **2021 cow dataset from the University of Bristol** as the initial development dataset.

The purpose of this phase is to:

- Understand the dataset
- Explore image characteristics
- Develop preprocessing
- Experiment with cattle segmentation
- Develop feature-extraction methods
- Establish initial regression models
- Identify limitations before collecting new smartphone data

The Bristol dataset is therefore an **initial development/experimentation dataset**, not the final BAIF deployment dataset.

## Tasks

1. Obtain and organize the dataset.
2. Study its documentation and image structure.
3. Inspect image resolution, viewpoints, backgrounds and animal poses.
4. Identify which images are suitable for the intended workflow.
5. Develop image preprocessing.
6. Develop/experiment with cattle segmentation.
7. Extract body-related features.
8. Establish baseline regression models.
9. Evaluate initial predictions.
10. Document differences between the Bristol dataset and the intended smartphone/BAIF environment.

---

# 6. Phase 2 — Smartphone Data Collection

After the initial pipeline is working, collect approximately:

> **50–100 real dairy cattle**

using a smartphone.

Each animal should have:

```text
Animal ID
+
Smartphone Image(s)
+
Actual Weight
+
Image-Capture Metadata
```

The actual weight should be obtained from a weighing scale and used as the ground-truth value.

---

# 7. Why a Standardized Image-Capture Protocol Is Required

A major risk in this project is inconsistent image acquisition.

The same cow can appear very different depending on:

- Camera distance
- Camera height
- Camera angle
- Cow orientation
- Cow posture
- Lighting
- Background
- Camera zoom
- Perspective distortion

If these factors vary too much, the model may learn image/camera conditions rather than genuine relationships between cattle morphology and body weight.

Therefore, the smartphone dataset must follow a **standardized image-capture protocol**.

The objective is not to force the animal to remain perfectly motionless. Instead:

> **The initial image-capture protocol should standardize camera distance, camera orientation, animal view and acceptable posture while allowing normal minor animal movement.**

---

# 8. Proposed Image-Capture Protocol

The exact numerical values should be finalized during a small pilot test before collecting the full 50–100 animals.

## 8.1 Primary View — Lateral / Side View

The primary image should be a **side/lateral view** of the cattle.

Conceptually:

```text
                 Camera
                   |
                   |
                   ↓

        HEAD                 RUMP
          ↓                    ↓
       ┌─────────────────────────┐
       │           COW           │
       └─────────────────────────┘

              ← Body Length →
```

The camera should be approximately perpendicular to the animal's side.

This view is preferred because it provides useful information for:

- Body length
- Body height
- Body silhouette
- Body area

## 8.2 Left and Right Side

Ideally, collect both:

- Left-side image
- Right-side image

However, one side should be defined as the **primary standardized orientation** for the initial model if necessary.

Both sides can later be compared to determine whether left/right orientation affects performance.

## 8.3 Camera Distance

The camera distance must be standardized or maintained within a defined acceptable range.

Do not collect images with arbitrary distances such as:

```text
Cow 1 → 1 metre
Cow 2 → 4 metres
Cow 3 → 0.8 metres
```

because apparent pixel dimensions change with distance.

The exact distance should be finalized through a pilot test based on:

- Smartphone field of view
- Cow size
- Image resolution
- Ability to fit the entire animal in the frame
- Perspective distortion

The final protocol should document the selected distance/range.

## 8.4 Camera Height

The camera should be positioned approximately around the middle/torso region of the animal rather than strongly above or below it.

Avoid large vertical camera angles because they introduce perspective distortion.

## 8.5 Camera Orientation

The camera should be approximately perpendicular to the side of the cattle for the primary image.

Avoid strongly angled photographs.

Preferred:

```text
Camera
  |
  |
  ↓
Cattle
──────────────
```

Avoid:

```text
Camera
   \
    \
     Cattle
```

## 8.6 Digital Zoom

Digital zoom should preferably not be used.

Instead:

1. Move to the standardized capture distance.
2. Position the animal correctly in the frame.
3. Ensure the entire animal is visible.
4. Capture the image.

## 8.7 Framing

The entire animal should be visible.

The frame should contain enough margin around the cattle to avoid cutting off:

- Head
- Legs
- Tail
- Back
- Rump

At the same time, the animal should occupy a sufficiently large portion of the image to preserve useful visual information.

## 8.8 Animal Pose

The system should not require a perfectly static pose.

### Preferable

- Naturally standing cattle
- Small natural head movement
- Slight leg movement
- Natural tail movement
- Minor body movement

### Avoid

- Sitting/lying cattle
- Severe bending
- Walking directly toward the camera
- Walking directly away
- Severe body rotation
- Heavy occlusion
- Another animal covering part of the target animal
- Only part of the body visible

The aim is:

> **Natural standing posture with controlled but realistic variation.**

---

# 9. Image Capture Metadata

Every photograph should ideally be associated with metadata.

Recommended fields:

```text
cow_id
image_id
actual_weight_kg
breed
age
capture_date
camera_model
camera_distance
camera_height
view
animal_orientation
pose
lighting_condition
ground_condition
image_quality
```

Not every field must necessarily become a model feature.

The metadata is useful for:

- Quality control
- Dataset analysis
- Error analysis
- Reproducibility
- Studying failure conditions

For example, it may later reveal that the model performs poorly when the cow is heavily rotated or when lighting is low.

---

# 10. Recommended Dataset Structure

A possible structure is:

```text
cattle_dataset/
│
├── images/
│   ├── C001/
│   │   ├── left_01.jpg
│   │   ├── left_02.jpg
│   │   ├── right_01.jpg
│   │   └── ...
│   │
│   ├── C002/
│   │   ├── left_01.jpg
│   │   └── ...
│   │
│   └── ...
│
├── annotations/
│   └── segmentation/
│
└── metadata.csv
```

Example metadata:

```csv
cow_id,image_id,weight_kg,view,distance,pose
C001,C001_L01,412,left,STANDARD,standing
C001,C001_R01,412,right,STANDARD,standing
C002,C002_L01,365,left,STANDARD,standing
```

The actual weight is associated with the **animal**, while multiple images may belong to the same animal.

---

# 11. Ground Truth Collection

The model needs reliable ground-truth weight.

For each animal:

```text
Cattle
  ↓
Actual weighing-scale measurement
  ↓
Weight in kg
  ↓
Stored against Animal ID
```

Example:

```text
Animal ID: C001
Actual Weight: 412 kg
Image: C001_L01.jpg
```

The photograph and weight should be collected as close together in time as practically possible so that the image represents the animal at the same physical condition as the ground-truth measurement.

---

# 12. Phase 3 — Image Preprocessing

After collecting images, preprocessing will be performed.

Potential operations include:

- Image resizing
- Image normalization
- Quality checking
- Removing unusable images
- Standardizing image format
- Handling excessive noise
- Maintaining consistent image dimensions

The exact preprocessing pipeline should be determined after examining the Bristol and smartphone datasets.

---

# 13. Phase 4 — Cattle Segmentation

The next major stage is separating the animal from the background.

```text
Original Image
      ↓
Segmentation Model
      ↓
Cattle Mask
      ↓
Cow Silhouette
```

Conceptually:

```text
Original:

🌳 🌾 🐄 🌾 🌳

After segmentation:

       🐄
```

The segmentation model should identify the cattle body while excluding:

- Background
- Trees
- Buildings
- People
- Other animals
- Unrelated objects

The segmentation stage is critical because the subsequent morphometric measurements depend on the quality of the cattle silhouette.

---

# 14. Phase 5 — Morphometric Feature Extraction

From the segmented cattle image, the project proposes extracting:

### 1. Body Length

Approximate longitudinal dimension of the animal.

```text
HEAD ←──────────────→ RUMP
       BODY LENGTH
```

### 2. Body Height

Approximate vertical body dimension.

```text
       BACK
        │
        │ HEIGHT
        │
       GROUND
```

### 3. Body Girth

Circumference-related body measurement.

This is technically more challenging from a single RGB image because a photograph does not directly provide 3D circumference.

Therefore, the exact method for estimating girth must be experimentally developed and validated.

### 4. Body Area

Area occupied by the segmented animal silhouette.

```text
Cow Mask
   ↓
Pixel area
   ↓
Body-area feature
```

The project documentation identifies length, height, girth and area as the main proposed features.

---

# 15. Critical Issue — Image Scale and Calibration

A smartphone image contains pixels, not direct centimetre measurements.

For example:

```text
Cow length = 650 pixels
```

does not automatically mean:

```text
650 pixels = 210 cm
```

unless the image contains enough information to establish scale.

Therefore, the project must investigate **image calibration / standardized capture / scale estimation**.

Possible research directions include:

- Standardized camera distance
- Standardized camera height
- Camera calibration
- Known reference dimensions
- Controlled capture geometry
- Other image-derived calibration methods

The final method should be selected after experimentation.

This is one of the most important technical challenges in the project.

---

# 16. Phase 6 — Weight Prediction Model

After feature extraction:

```text
Body Length ──────┐
Body Height ──────┤
Body Girth ───────┼──→ Regression Model → Predicted Weight
Body Area ────────┘
```

The project documents do not require one specific final regression algorithm.

Therefore, several approaches can be compared.

Possible candidates include:

- Linear Regression
- Random Forest Regressor
- Support Vector Regression
- Gradient Boosting
- XGBoost
- Multi-Layer Perceptron / ANN
- Other appropriate regression models

The final model should be selected based on validation performance and suitability for the small pilot dataset.

---

# 17. Why Model Comparison Is Important

With only approximately 50–100 animals, using a very large deep-learning regression network could result in overfitting.

Therefore, the project should establish baseline models first.

Recommended experimental progression:

```text
Linear Regression
       ↓
Random Forest
       ↓
Gradient Boosting / XGBoost
       ↓
SVR
       ↓
MLP / Deep Learning
       ↓
Compare performance
       ↓
Select best model
```

This also makes the research more defensible because the project can demonstrate why the selected model performs better than simpler alternatives.

---

# 18. Train / Validation / Test Strategy

The dataset should be separated into:

```text
Training Set
      ↓
Model learns relationship

Validation Set
      ↓
Hyperparameter/model selection

Test Set
      ↓
Final unbiased evaluation
```

A critical point is that images of the **same animal should not be split across training and testing sets**.

For example, this is problematic:

```text
C001_left.jpg → Training
C001_right.jpg → Testing
```

because the model may learn the individual animal rather than generalize to unseen animals.

Instead, split at the **animal level**:

```text
Training animals
Validation animals
Testing animals
```

This is especially important when multiple images are collected per cow.

---

# 19. Model Evaluation

The final model should be evaluated against actual weighing-scale measurements.

Important metrics include:

## MAE — Mean Absolute Error

Average absolute difference between predicted and actual weight.

Example:

```text
Actual:    400 kg
Predicted: 420 kg
Error:      20 kg
```

## RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

## R² — Coefficient of Determination

Measures how much of the variance in actual weight is explained by the model.

## Correlation

Measure the strength of association between predicted and actual weights.

The project currently targets a strong correlation, broadly described as approximately **80–90%**, but this should remain a target rather than a guaranteed result.

The actual achieved performance should be reported honestly after testing.

---

# 20. Predicted vs Actual Weight Analysis

A key visualization should be:

```text
Actual Weight
      ↑
      │        •
      │      •
      │    •
      │  •
      │ •
      └────────────────→ Predicted Weight
```

A strong model should show predictions close to the ideal relationship.

Other useful analyses include:

- Residual plots
- Error distribution
- MAE/RMSE comparison
- Performance by animal size
- Performance by breed, if sufficient data exists
- Performance by image condition
- Performance by pose
- Performance by capture angle

---

# 21. Phase 7 — Smartphone-Based Prototype

Once the model is sufficiently reliable, integrate it into a simple web interface.

Proposed workflow:

```text
Field Worker
     ↓
Upload Cattle Image
     ↓
Backend
     ↓
Image Preprocessing
     ↓
Cattle Segmentation
     ↓
Feature Extraction
     ↓
Weight Prediction
     ↓
Estimated Weight
```

Example interface:

```text
┌─────────────────────────────────────────┐
│      BAIF CATTLE WEIGHT ESTIMATOR       │
├─────────────────────────────────────────┤
│                                         │
│          Upload Cattle Image             │
│                                         │
│           [ Choose Image ]              │
│                                         │
│              [ Predict ]                │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│          Estimated Weight               │
│                                         │
│              417.6 kg                   │
│                                         │
└─────────────────────────────────────────┘
```

The field worker should not need to understand the underlying computer-vision pipeline.

The intended user experience is:

> **Image → Prediction**

---

# 22. Proposed Technology Stack

## Programming

- Python

## Computer Vision

- OpenCV

## Deep Learning

- PyTorch

## Machine Learning

- Scikit-learn

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

## Development

- VS Code
- Jupyter Notebook / JupyterLab
- PyCharm

## Version Control

- Git

## Deployment

- Web-based application
- Laptop or cloud-based computing environment

The project documentation identifies OpenCV, PyTorch, Scikit-learn, Pandas, NumPy and Matplotlib as the core software ecosystem.

---

# 23. Hardware Requirements

The project can be developed using a normal academic/personal computer.

Minimum-oriented configuration:

- Intel Core i5 / AMD Ryzen 5 or equivalent
- 8 GB RAM
- 20–30 GB free storage
- Integrated graphics for basic processing
- GPU recommended for deep-learning training
- Windows/Linux/macOS
- Smartphone with a functional camera

Recommended:

- Intel Core i7 / AMD Ryzen 7 or higher
- 16–32 GB RAM
- SSD
- NVIDIA GPU or cloud GPU
- Good-quality smartphone camera

---

# 24. Phase 8 — BAIF Dataset

After the initial prototype has been developed and evaluated using the Bristol and locally collected smartphone data, the project will move toward the **BAIF dataset**.

The purpose of the BAIF data is to evaluate and improve the system under the real environment for which it is ultimately intended.

Potential objectives:

- Test generalization
- Identify field-specific failure cases
- Improve segmentation
- Improve calibration
- Improve weight prediction
- Evaluate performance on real BAIF cattle
- Understand environmental variation
- Prepare for practical deployment

The exact BAIF dataset size, format, metadata and collection conditions should be determined with BAIF before implementation.

---

# 25. Bristol → Smartphone → BAIF Strategy

The three datasets serve different purposes.

| Dataset | Main Purpose |
|---|---|
| University of Bristol 2021 | Initial development and experimentation |
| 50–100 personally collected smartphone cattle | Adaptation, pilot validation and real smartphone testing |
| BAIF dataset | Real-world validation and improvement |

This should not be treated as one homogeneous dataset.

Differences between datasets should be explicitly analyzed.

---

# 26. Dataset Domain Shift

A major issue will be **domain shift**.

For example:

```text
Bristol Dataset
     ↓
Different camera
Different environment
Different breeds
Different image conditions
     ↓
Smartphone Dataset
     ↓
Indian field conditions
     ↓
BAIF Dataset
```

A model that performs well on Bristol images may not automatically perform equally well on Indian smallholder cattle.

Therefore, one important research question will be:

> **How well does a model developed using an external dataset transfer to smartphone images captured under Indian field conditions?**

The 50–100 animal smartphone dataset acts as an important bridge between the initial dataset and the eventual BAIF environment.

---

# 27. Data Augmentation

If appropriate, training data can be augmented using techniques such as:

- Small rotations
- Horizontal flipping where valid
- Cropping
- Brightness variation
- Contrast variation
- Mild scaling
- Other realistic transformations

Augmentation should not create unrealistic cattle geometry.

The purpose is to improve robustness to normal image variation without changing the underlying physical relationship.

---

# 28. Image-Based Phase vs Video-Based Phase

## Phase 1 — Image-Based

The initial project should remain image-based.

```text
One image
   ↓
Segmentation
   ↓
Feature extraction
   ↓
Weight prediction
```

This is easier to develop, evaluate and debug.

## Phase 2 — Video-Based

After the image-based prototype is working properly, move toward video.

```text
Video
 ↓
Frame extraction
 ↓
Cattle detection/segmentation
 ↓
Feature extraction
 ↓
Frame-level estimates
 ↓
Temporal aggregation
 ↓
Final weight estimate
```

---

# 29. Why Video Is a Natural Future Extension

A real cow will not always stand perfectly still.

During video capture it may:

- Move its head
- Move its legs
- Move its tail
- Walk slowly
- Change body orientation
- Shift posture

A video system can use multiple frames instead of depending on one photograph.

Potentially:

```text
Frame 1 → 410 kg
Frame 2 → 415 kg
Frame 3 → 413 kg
Frame 4 → 418 kg
Frame 5 → 414 kg

          ↓
   Temporal aggregation

      Final estimate
          414 kg
```

The exact video methodology will be designed after the image-based system is validated.

---

# 30. Important Image-Based vs Video-Based Principle

The video system should **not simply be the image model running independently on every frame** without further analysis.

The future video phase should investigate:

- Frame selection
- Temporal consistency
- Duplicate-frame handling
- Motion blur
- Pose variation
- Aggregation of frame-level predictions
- Confidence estimation
- Stable final weight prediction

---

# 31. Scope of the Current Project

## Included

- Dairy cows
- Smartphone images
- Computer vision
- Cattle segmentation
- Morphometric feature extraction
- Weight regression
- Approximately 50–100 pilot animals
- Actual weighing-scale ground truth
- Model validation
- Web-based prototype
- Initial external dataset development using the Bristol dataset
- Later BAIF validation

## Not Included in the Initial Phase

- Buffalo weight estimation
- Other livestock species
- 3D cameras
- Depth cameras
- RGB-D sensors
- Very large datasets involving thousands of animals
- Certified/legal replacement of weighing scales
- Continuous real-time weight monitoring
- IoT-based weighing systems
- Large-scale commercial deployment
- Dedicated Android/iOS application
- Full-scale field deployment beyond the pilot stage

These may be future extensions.

---

# 32. Key Research Challenges

## Challenge 1 — Small Dataset

50–100 animals is a relatively small dataset for deep learning.

**Response:**

- Use transfer learning where appropriate
- Compare classical ML models
- Use augmentation carefully
- Split data at animal level
- Avoid overly complex models
- Report limitations

## Challenge 2 — Pixel-to-Physical Measurement

A smartphone image gives pixel measurements, not direct centimetre measurements.

**Response:**

- Standardize capture geometry
- Investigate calibration
- Test known reference/scale approaches if feasible
- Evaluate how distance affects extracted measurements

## Challenge 3 — Animal Movement

Cattle do not always remain perfectly still.

**Response:**

- Permit natural minor movement in image capture
- Define acceptable poses
- Later transition to video-based estimation

## Challenge 4 — Background Variation

Real farms have complex backgrounds.

**Response:**

- Train/validate segmentation under realistic conditions
- Collect varied field backgrounds
- Evaluate segmentation failure cases

## Challenge 5 — Domain Shift

Bristol data may differ substantially from Indian/BAIF data.

**Response:**

- Use Bristol for initial development
- Collect 50–100 Indian smartphone samples
- Fine-tune/adapt models
- Evaluate on BAIF data later

## Challenge 6 — Limited Ground Truth

Obtaining accurate actual weights requires access to weighing equipment.

**Response:**

- Pair every collected animal with an actual scale measurement
- Maintain consistent data recording
- Use actual weight as ground truth

---

# 33. Recommended Experimental Roadmap

## Experiment 1 — Dataset Exploration

Analyze the Bristol dataset:

- Number of images
- Image dimensions
- Viewpoints
- Backgrounds
- Animal poses
- Available weight information
- Data quality

## Experiment 2 — Segmentation

Test suitable segmentation approaches.

Evaluate:

- Mask quality
- Boundary accuracy
- Robustness to background

## Experiment 3 — Feature Extraction

Test:

- Length
- Height
- Area
- Girth estimation

Evaluate whether these features are stable under changes in:

- Distance
- Angle
- Pose
- Lighting

## Experiment 4 — Baseline Regression

Train:

- Linear Regression
- Random Forest
- SVR
- Gradient Boosting/XGBoost

## Experiment 5 — Deep Learning Regression

Test an MLP or another appropriate deep-learning regression approach if justified by the data.

## Experiment 6 — Smartphone Pilot

Collect 50–100 real animals.

Test the complete pipeline.

## Experiment 7 — Cross-Dataset Evaluation

Compare:

```text
Bristol-trained model
        ↓
Indian smartphone images
```

and study performance degradation.

## Experiment 8 — Adaptation

Fine-tune/retrain using the smartphone dataset.

## Experiment 9 — Prototype

Deploy the best validated pipeline in a web interface.

## Experiment 10 — BAIF Validation

Test and improve the system using BAIF data.

## Experiment 11 — Video Extension

Only after the image-based system is stable.

---

# 34. Suggested Project Timeline

The exact dates depend on BAIF's schedule and dataset availability, but the logical order is:

### Stage 1 — Initial Setup

- Obtain Bristol 2021 dataset
- Study dataset documentation
- Set up Python environment
- Organize repository
- Perform EDA

### Stage 2 — Initial Computer Vision

- Image preprocessing
- Segmentation experiments
- Silhouette generation
- Feature extraction

### Stage 3 — Initial ML Model

- Prepare feature table
- Train baseline regression models
- Compare metrics
- Identify limitations

### Stage 4 — Smartphone Capture Protocol

Before collecting 50–100 cows:

- Finalize camera position
- Finalize approximate distance/range
- Finalize side-view protocol
- Define acceptable pose
- Define metadata
- Test protocol on a small number of animals

### Stage 5 — 50–100 Cow Dataset

- Collect smartphone images
- Collect ground-truth weight
- Store metadata
- Perform quality control

### Stage 6 — Model Adaptation

- Preprocess new data
- Train/fine-tune segmentation
- Extract features
- Train regression models
- Compare results

### Stage 7 — Validation

- Hold-out test animals
- Calculate MAE
- Calculate RMSE
- Calculate R²
- Calculate correlation
- Analyze errors

### Stage 8 — Prototype

- Backend
- Image upload
- Model inference
- Weight display
- Basic user interface

### Stage 9 — BAIF Data

- Obtain BAIF dataset
- Understand its structure
- Evaluate model
- Identify domain shift
- Improve model

### Stage 10 — Final Image Prototype

- Freeze/validate model
- Document methodology
- Prepare demonstrations
- Prepare results

### Stage 11 — Future Video Phase

- Collect sample videos
- Extract frames
- Evaluate image model frame-by-frame
- Investigate temporal aggregation
- Develop video-based prototype if time/resources permit

---

# 35. Recommended Repository Structure

```text
cattle-weight-estimation/
│
├── data/
│   ├── bristol/
│   ├── smartphone_pilot/
│   ├── baif/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_segmentation.ipynb
│   ├── 04_feature_extraction.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_deep_learning.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_error_analysis.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── inference/
│
├── models/
│
├── web/
│
├── configs/
│
├── results/
│
├── reports/
│
├── requirements.txt
│
└── README.md
```

---

# 36. Data Quality Checklist

Before accepting an image into the dataset, check:

- [ ] Entire cow visible
- [ ] Correct target animal
- [ ] Animal not heavily occluded
- [ ] Primary side view approximately achieved
- [ ] Camera distance within protocol
- [ ] Camera orientation acceptable
- [ ] Image is sharp enough
- [ ] Lighting is sufficient
- [ ] No excessive motion blur
- [ ] No severe perspective distortion
- [ ] Animal pose is acceptable
- [ ] Correct Animal ID recorded
- [ ] Actual weight recorded
- [ ] Metadata recorded

---

# 37. Model Quality Checklist

Before declaring the prototype successful:

- [ ] Segmentation works on unseen animals
- [ ] Features can be extracted consistently
- [ ] Train/validation/test split is animal-level
- [ ] No data leakage
- [ ] Baseline models evaluated
- [ ] Deep-learning approach evaluated where appropriate
- [ ] MAE calculated
- [ ] RMSE calculated
- [ ] R² calculated
- [ ] Correlation calculated
- [ ] Predicted vs actual graph created
- [ ] Error distribution analyzed
- [ ] Poor predictions investigated
- [ ] Smartphone-domain performance tested
- [ ] Prototype tested with unseen images

---

# 38. Important Success Criteria

The project should not define success only as "80–90% correlation."

Success should include:

### Technical

- Reliable cattle segmentation
- Stable feature extraction
- Meaningful relationship between features and weight
- Good regression performance

### Research

- Proper experimental design
- No data leakage
- Clear comparison of models
- Transparent evaluation
- Error analysis
- Clear limitations

### Practical

- Smartphone-only capture
- Simple field-worker workflow
- Fast prediction
- No manual measurement during routine use
- No specialized depth/3D camera

### Deployment

- Working web prototype
- Image upload
- Automatic processing
- Estimated weight output

The **80–90% correlation should be treated as a target**, not a guaranteed result.

---

# 39. What the Final Prototype Should Look Like

The desired user experience is:

```text
                FIELD WORKER
                     │
                     ▼
              Open Web Portal
                     │
                     ▼
             Upload Cow Image
                     │
                     ▼
              Automatic System
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Segmentation          Preprocessing
          │                     │
          └──────────┬──────────┘
                     ▼
             Feature Extraction
                     │
                     ▼
              Weight Prediction
                     │
                     ▼
              ┌─────────────┐
              │ Estimated   │
              │ Weight      │
              │ 417.6 kg    │
              └─────────────┘
```

The system should hide the complexity of the underlying ML pipeline from the field worker.

---

# 40. Future Scope

## 40.1 Buffaloes

Extend the system from dairy cows to buffaloes.

## 40.2 Larger Dataset

Increase the dataset beyond 50–100 animals.

Include variation in:

- Breed
- Age
- Body condition
- Farm
- Geography
- Environment

## 40.3 Improved Accuracy

Experiment with:

- Better segmentation
- Additional features
- Advanced regression
- Deep-learning approaches
- Better calibration

## 40.4 Dedicated Mobile Application

Eventually develop an Android/iOS application.

## 40.5 Large-Scale Field Deployment

Test the system across multiple farms and locations.

## 40.6 Livestock Management Integration

Integrate estimated weight with livestock records for:

- Feeding
- Health
- Breeding
- Growth monitoring

## 40.7 Video-Based Estimation

Use video to handle:

- Natural movement
- Multiple poses
- Multiple frames
- Temporal consistency

---

# 41. Final End-to-End Architecture

```text
                 ┌─────────────────────┐
                 │ Smartphone Camera   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Image Acquisition   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Preprocessing       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Cattle Segmentation │
                 │ Deep Learning       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Cattle Silhouette   │
                 └──────────┬──────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │ Morphometric Feature         │
             │ Extraction                   │
             │                              │
             │ Length                       │
             │ Height                       │
             │ Girth                        │
             │ Area                         │
             └──────────────┬───────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Regression Model    │
                 │ ML / DL             │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Estimated Weight    │
                 │       (kg)          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Web-Based Portal    │
                 └─────────────────────┘
```

---

# 42. Final Project Roadmap

```text
                         START
                           │
                           ▼
                Bristol 2021 Dataset
                           │
                           ▼
                  Dataset Exploration
                           │
                           ▼
                 Initial CV Pipeline
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
        Segmentation              Feature Extraction
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  Initial Regression
                           │
                           ▼
                 Pilot Smartphone Study
                           │
                           ▼
                 50–100 Real Cattle
                           │
                           ▼
              Standardized Image Capture
                           │
                           ▼
                 Ground Truth Weights
                           │
                           ▼
                 Model Adaptation
                           │
                           ▼
                    Evaluation
                           │
                           ▼
              Working Image Prototype
                           │
                           ▼
                     BAIF Dataset
                           │
                           ▼
             Real-World Validation
                           │
                           ▼
                Improved Final Model
                           │
                           ▼
                  Web-Based Prototype
                           │
                           ▼
                 Future Video System
                           │
                           ▼
                          END
```

---

# 43. Important Note on the Existing Project Documents

The project documents correctly describe the cattle body-weight estimation workflow, but the current Synopsis contains an **incorrect methodology section** from an earlier project.

The section under "Methodology / Implementation Plan" currently contains references to:

- TCGA-GBM
- BraTS
- Federated Learning
- XGBoost/Cox
- Hospital nodes
- FedAvg
- Kaplan-Meier curves
- Survival analysis

Those items do not belong to the cattle body-weight project and should be replaced with the cattle-specific methodology described in this document.

The correct methodology should focus on:

```text
Bristol Dataset
      ↓
Image Preprocessing
      ↓
Cattle Segmentation
      ↓
Morphometric Feature Extraction
      ↓
Regression
      ↓
Smartphone Pilot Dataset
      ↓
Model Adaptation
      ↓
Validation
      ↓
BAIF Dataset
      ↓
Web Prototype
      ↓
Future Video Extension
```

---

# 44. One-Sentence Project Definition

> **This project develops a low-cost computer-vision system that estimates dairy cattle body weight from standardized smartphone images by segmenting the animal, extracting morphometric features, and applying a trained regression model, with the eventual goal of validating the system using BAIF field data and extending it from single-image estimation to video-based estimation.**

---

# 45. Final Understanding

The project is therefore **not simply a cattle image-classification project**.

It is an end-to-end research and prototype-development pipeline:

**Data → Computer Vision → Morphometry → Machine Learning → Weight Prediction → Validation → Web Deployment → BAIF Field Validation → Future Video Extension**

The most important things to get right are:

1. **Reliable ground-truth weights**
2. **Standardized smartphone image acquisition**
3. **Correct cattle segmentation**
4. **Reliable image-to-physical measurement/calibration**
5. **Animal-level train/test separation**
6. **Model comparison**
7. **Realistic evaluation**
8. **Domain adaptation from Bristol → Indian smartphone data → BAIF**
9. **Simple field-worker workflow**
10. **Video extension only after the image-based system is stable**

