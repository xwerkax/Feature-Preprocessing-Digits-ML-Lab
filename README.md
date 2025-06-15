## Lab 4 – Feature Preprocessing and Engineering

This repository contains solutions to Lab 4 of a Machine Learning course. The goal is to evaluate different classification models on a real-world dataset, both with and without feature preprocessing.

## 🧠 Overview

The experiment uses the **Digits dataset** from `scikit-learn`, which contains low-resolution images of handwritten digits. Each image is represented as a vector of pixel intensities.

We compare the performance of 3 classifiers:
- Gaussian Naive Bayes (GNB)
- k-Nearest Neighbors (KNN)
- Decision Tree (DT)

## ✅ Task 1: Baseline (Raw Features)

- Dataset: `sklearn.datasets.load_digits()`
- Cross-validation: **5 repetitions**, **2-fold CV** (total of 10 evaluations per model)
- Models: GNB, KNN, DT
- Metrics:
  - Mean accuracy and standard deviation for each model
  - Results rounded to 3 decimal places

 ## ✅ Task 2: Standardized Features (Preprocessing with StandardScaler)

- Applied `StandardScaler` to normalize features:
  - `fit` on training data,
  - `transform` applied to both training and test sets in each fold.
- Cross-validation identical to Task 1.
- Same metrics reported.
