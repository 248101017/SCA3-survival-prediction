# SCA3 Survival Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sca3-survival-prediction-nafxd6esutvfhdvfpzyn92.streamlit.app/)

Machine learning-based survival prediction in spinocerebellar ataxia type 3 (SCA3) using DeepSurv, random survival forest, deep Cox mixtures, and Cox proportional hazards models.

## Overview

This repository contains the source code, pre-trained model artifacts, and a web-based clinical prediction tool accompanying the manuscript:

> **Machine Learning-Based Survival Prediction in Spinocerebellar Ataxia Type 3: Development, Validation, and Neuroimaging Corroboration**
>
> Ziting Cui, Jian Hu, Rongfan Peng, Yun Peng, et al.
>
> *npj Digital Medicine* (submitted)

## Repository structure

SCA3-survival-prediction/
├── app.py                     # Web-based prediction tool (Streamlit)
├── scripts/
│   ├── data_preprocessing.py  # Data cleaning, imputation, correlation matrix
│   ├── lasso_cox.R            # LASSO–Cox variable selection
│   ├── model_training.py      # Model training, evaluation, SHAP, bootstrap
│   ├── external_inference.py  # External cohort risk score inference
│   └── scatter_vbm.py         # VBM scatter plot analysis
├── artifacts/                 # Pre-trained model files
│   ├── deepsurv_dcph_model.pkl
│   ├── rsf_auton_model.pkl
│   ├── transformer.pkl
│   ├── times.pkl
│   ├── feature_ranges.json
│   └── cutoffs_*.json
├── requirements.txt           # sca3infer (web app) dependencies
├── requirements_full.txt      # sca3train (training pipeline) dependencies
├── runtime.txt                # Streamlit Cloud Python runtime pin
├── LICENSE
└── README.md

## Web-based clinical prediction tool

The DeepSurv model is deployed as a Streamlit web application that accepts 10 clinical predictors and returns individualised survival probability estimates at approximately 5 and 9 years, together with a predicted survival curve and risk-group classification.

### Online access (Streamlit Cloud)
https://sca3-survival-prediction-nafxd6esutvfhdvfpzyn92.streamlit.app/

### Local deployment

git clone https://github.com/248101017/SCA3-survival-prediction.git
cd SCA3-survival-prediction
pip install -r requirements.txt
streamlit run app.py

## Input features (DeepSurv)

The web tool requires the following 10 baseline clinical predictors:

| Feature | Type | Description |
|---|---|---|
| SARA score | Continuous | Scale for the Assessment and Rating of Ataxia |
| Disease duration | Continuous | Years since symptom onset |
| BMI | Continuous | Body mass index (kg/m²) |
| Long CAG repeats | Continuous | ATXN3 expanded CAG repeat length |
| EQ-VAS | Continuous | EuroQol visual analogue scale (0–100) |
| PHQ-9 depression | Continuous | Patient Health Questionnaire-9 score |
| GAD-7 anxiety | Continuous | Generalized Anxiety Disorder-7 score |
| INAS muscle atrophy | Binary (0/1) | Inventory of Non-Ataxia Signs |
| INAS fasciculations | Binary (0/1) | Inventory of Non-Ataxia Signs |
| INAS sensory symptoms | Binary (0/1) | Inventory of Non-Ataxia Signs |

## Data availability

Individual-level clinical data are not publicly available due to patient privacy and ethical restrictions, and are not included in this repository. De-identified summary statistics are provided in the manuscript supplementary tables. Requests for data access should be directed to the corresponding author.

## Environment setup

Two conda environments were used in this study:

- sca3infer (web application): dependencies are listed in requirements.txt.
- sca3train (training/analysis pipeline): dependencies are listed in requirements_full.txt.

To reproduce the web application only:

pip install -r requirements.txt
streamlit run app.py

To reproduce the full analysis pipeline:

pip install -r requirements_full.txt

## Software versions (key packages)

Key pinned packages for sca3infer (see requirements.txt for the complete list):

- streamlit==1.55.0
- numpy==1.26.4
- pandas==1.5.3
- matplotlib==3.10.8
- joblib==1.5.3
- scikit-learn==1.2.2
- auton-survival==0.1.0
- torch==1.13.1

Key pinned packages for sca3train (see requirements_full.txt for the complete list):

- numpy==1.26.4
- pandas==1.5.3
- scipy==1.10.1
- scikit-learn==1.2.2
- scikit-survival==0.21.0
- seaborn==0.13.2
- shap==0.49.1
- pyreadstat==1.3.3
- openpyxl==3.1.5

R was used for LASSO–Cox feature screening:

- R==4.5.1
- glmnet (R) version: 4.1-10 (as used in this study)

Neuroimaging analyses were performed using:

- SPM12
- CAT12
- MATLAB (R2023a)

## Reproducing the analysis pipeline

The raw clinical data are not included in this repository. The following commands are provided for reproducibility in approved environments.

### Step 1: Data preprocessing

python scripts/data_preprocessing.py --input data/data.sav --outdir outputs/lasso

### Step 2: LASSO–Cox variable selection

Rscript scripts/lasso_cox.R --input outputs/lasso/features_imputed_academic.csv --outdir outputs/lasso

### Step 3: Model training and evaluation

python scripts/model_training.py --input data/processed_cohort.csv --outdir outputs --artifacts artifacts

### Step 4: External cohort inference

python scripts/external_inference.py --input data/external_cohort.xlsx --artifacts artifacts --outdir outputs

### Step 5: VBM scatter plot analysis

python scripts/scatter_vbm.py --clinical data/external_cohort.xlsx --risk outputs/outer_risk_predictions.csv --outdir outputs/scatter

## License

This project is licensed under the MIT License. See LICENSE for details.

## Contact

For questions regarding the code, model, or data access, please open a GitHub Issue or contact the corresponding author.