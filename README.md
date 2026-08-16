# Cattle Body Weight Estimation using Computer Vision

This project implements a non-contact, smartphone-based system for estimating the body weight of dairy cattle from side-view images, specifically tailored for Indian smallholder dairy farming conditions.

## Project Structure

- `data/`: Datasets for training, validation, and testing.
  - `bristol/`: University of Bristol 2021 dataset (initial development).
  - `smartphone_pilot/`: 50-100 smartphone images + ground-truth weights.
  - `baif/`: Real-world field dataset from BAIF.
- `notebooks/`: Jupyter Notebooks for step-by-step experimentation (EDA, Segmentation, Regression).
- `src/`: Core Python modules for preprocessing, segmentation, feature extraction, and models.
- `models/`: Stored weights and serialized models.
- `configs/`: Hyperparameters and configuration files.
- `web/`: Frontend and backend for the web prototype.
- `results/`: Validation metrics, error distribution, plots, and figures.
- `reports/`: Documents and write-ups.

## Getting Started

1. Set up the Python environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the Bristol 2021 dataset in `data/bristol/`.
3. Open `notebooks/01_dataset_exploration.ipynb` to start exploring the dataset.
