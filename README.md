# Fraud Detection using Machine Learning

## Overview

This project presents a systematic and robust approach for detecting fraudulent financial transactions using the **IEEE-CIS Fraud Detection** dataset. The solution addresses challenges such as extreme class imbalance, high dimensionality, and missing values, employing advanced machine learning models and ensemble techniques for optimal performance.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Ensemble Learning](#ensemble-learning)
- [Evaluation Metrics](#evaluation-metrics)
- [Final Results](#final-results)


## Dataset

The **IEEE-CIS Fraud Detection Dataset**, made available via Kaggle, consists of:
- **590,540** transaction records (train set) with ~3.5% labeled as fraud.
- **434** anonymized features spanning transaction info, identity, device, card, email, and address.
- Test set includes **507,000+** records.

Challenges:
- Severe class imbalance.
- Extensive anonymization.
- Large amounts of missing data.

## Preprocessing

Key preprocessing steps included:
- **Memory Optimization:** Downcasted float64 and int64 data types, reducing memory usage by ~67%.
- **Missing Value Handling:** Missing entries imputed with `-999` (models treat missingness as a signal).
- **Minimal Categorical Encoding:** Tree-based models (LightGBM, CatBoost) inherently handle categorical features.
- **Feature Consistency Checks:** Outliers and low-variance features were reviewed.
- **Stratified Train-Validation Split:** 80/20 split maintaining class ratio.

> Future enhancements may include time-based cross-validation due to temporal drift.

## Models Used

A variety of models were trained:

### Baseline Models:
- **Logistic Regression**
- **Random Forest**
- **Scikit-learn Gradient Boosting**

### Advanced Boosting Models:
- **XGBoost** – efficient with missing data and regularization.
- **CatBoost** – native handling of categorical features.
- **LightGBM** – leaf-wise growth, histogram-based optimization.

> Hyperparameter tuning was conducted using **Optuna** with Bayesian optimization.

## Ensemble Learning

To maximize performance, an ensemble was built by **averaging predictions** from:
- XGBoost
- CatBoost
- LightGBM

> **Weighted voting** was used to leverage model diversity and improve generalizability.

## Evaluation Metrics

Due to class imbalance, the following metrics were emphasized:
- **AUC-ROC** (Primary Metric)
- **Precision**
- **Recall**
- **F1 Score**

Macro-averaging was used to ensure fair evaluation across both classes.

| Model                    | Precision | Recall | F1 Score |
|---------------------------|-----------|--------|----------|
| LightGBM                  | 0.97      | 0.82   | 0.88     |
| XGBoost                   | 0.96      | 0.74   | 0.81     |
| CatBoost                  | 0.96      | 0.74   | 0.82     |
| Ensemble (XGB + Cat + LGBM)| 0.97      | 0.82   | 0.88     |

## Final Results

| Model                    | Test AUC  |
|---------------------------|-----------|
| LightGBM                  | 0.932419  |
| XGBoost                   | 0.928005  |
| CatBoost                  | 0.925913  |
| Ensemble (XGB + Cat + LGBM)| **0.932421** |

> The ensemble achieved the highest AUC, though LightGBM was nearly identical. Ensemble learning slightly improved robustness against variance.
