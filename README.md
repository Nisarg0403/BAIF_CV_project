# Computer Vision-Based Cattle Body Weight Estimation

This project implements a non-contact, smartphone-based system for estimating the body weight of dairy cattle from side-view images, specifically tailored for Indian smallholder dairy farming conditions.

---

## 📋 What the Project is About

Measuring cattle body weight is essential for feeding, breeding, and veterinary health monitoring. Traditional scales are expensive, unavailable in remote areas, or require stressful physical handling. 

This system provides a low-cost, practical alternative using a standard smartphone:
1. **Image Acquisition**: Capture a standardized lateral (side-view) image of a cow.
2. **Cattle Segmentation**: A pre-trained deep learning model (**DeepLabV3-ResNet50**) separates the cow from the surrounding background.
3. **Morphometric Feature Extraction**: The system calculates the cow's silhouette **Area**, **Body Length**, **Body Height**, and **Torso Girth** (surrogate torso depth) in pixels, calibrated into physical centimeters.
4. **Weight Regression**: An **XGBoost Regressor** predicts the cow's body weight (in kg) based on the extracted morphometric features using zoometric equations.

---

## 🛠️ Things Needed for the Project

### 1. Hardware Requirements
* **Smartphone**: A standard mobile phone with a working camera to take photos.
* **Computing Host**: A local laptop/PC or a cloud server (such as Streamlit Cloud) to run the PyTorch and machine learning models.

### 2. Software & Dependencies (Python)
The system requires **Python >= 3.8** and the following core libraries (listed in `requirements.txt`):
* **Deep Learning**: `torch`, `torchvision` (for the DeepLabV3 segmentation model).
* **Computer Vision**: `opencv-python`, `Pillow` (for image cropping, overlays, and transformations).
* **Machine Learning**: `xgboost`, `scikit-learn` (for weight prediction).
* **Web Dashboard**: `streamlit` (for the mobile-friendly web app).
* **Data Processing**: `numpy`, `pandas`, `matplotlib`, `seaborn` (for evaluation).

### 3. Image Capture Protocol (Criteria for Success)
To ensure accurate weight estimation, uploaded photos must follow these guidelines:
* **Lateral View**: The camera must be perpendicular to the side of the cow.
* **Consistent Distance**: Photos must be taken from a standard distance (e.g., 2.5 meters).
* **Full Body Visibility**: The entire cow (from head to tail, back to hooves) must be visible in the frame.
* **Natural Posture**: The cow must be standing naturally (no sitting, lying down, or extreme bending).

---

## 🚀 How to Run the Web Dashboard Locally

1. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/Nisarg0403/BAIF_CV_project.git
   cd BAIF_CV_project
   ```
2. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the models on the development dataset (to generate the regressor weights):
   ```bash
   python -m src.models.train
   ```
4. Start the Streamlit app:
   ```bash
   streamlit run web/app.py
   ```
5. Open **[http://localhost:8501](http://localhost:8501)** in your browser or phone to upload photos and estimate weights.
