## 🌿 Pistachio Species Classification with Neural Network
## 📌 Project Summary

This project implements a binary classification model to distinguish between Kirmizi and Siirt pistachio species using morphological features.

A Multi-Layer Perceptron (MLP) neural network was trained and evaluated with proper preprocessing and stratified splitting.

## 📊 Dataset

2148 samples

16 numerical morphological features

Binary target variable

Stratified 80/20 train-test split

## ⚙️ Methodology

Data cleaning & label encoding

Feature scaling using StandardScaler

MLPClassifier with hidden layers (32,16)

Model evaluation using accuracy, precision, recall, F1-score

Confusion matrix analysis

Model persistence using joblib

## 📈 Performance

Test Accuracy: 90.9%

Balanced precision and recall across both classes

Only 39 misclassifications out of 430 test samples

The model demonstrates strong class separability and stable generalization performance.

## 💾 Model Files

The trained model and scaler are stored in the models/ directory:

egitilmis_model.pkl
veri_olcekleme_araci.pkl


They can be loaded directly without retraining.

## 🛠 Technologies

Python

Pandas

Scikit-learn

Matplotlib

Seaborn

Joblib

## 🚀 Key Takeaway

Morphological features combined with neural network models provide highly effective discrimination between pistachio species.
