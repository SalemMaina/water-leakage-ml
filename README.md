# Water Leakage Detection — Machine Learning Pipeline

## Overview
A machine learning pipeline for classifying water pipeline leakages
into three categories:
- **0** — No Leak
- **1** — Minor Leak  
- **2** — Major Burst

## Methodology
- Statistical feature engineering on IoT sensor data
  (pressure and flow rate readings)
- Random Forest multi-class classifier
- SMOTE for class imbalance handling

## Dataset
LeakDB — Leakage Diagnosis Benchmark dataset
(Net1_CMH network, 1000 scenarios, 30-minute timesteps)

## Project Structure
water_leakage_ml/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   └── features.py
├── requirements.txt
└── README.md

## Requirements
Install dependencies with:
pip install -r requirements.txt

## Results
| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Leak | 0.93 | 0.98 | 0.95 |
| Minor Leak | 0.69 | 0.63 | 0.66 |
| Major Burst | 0.88 | 0.61 | 0.72 |

**Weighted F1: 0.896**

## Authors
Salem Maina
Final year project — JKUAT
