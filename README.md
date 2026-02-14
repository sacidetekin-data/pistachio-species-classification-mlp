🌿 Pistachio Species Classification using MLP Neural Network
📌 Project Overview

This project classifies two pistachio species — Kirmizi and Siirt — using morphological features and a Multi-Layer Perceptron (MLP) neural network.

The dataset contains 2148 samples with 16 numerical features extracted from pistachio images.

The goal is to build a robust binary classification model with strong generalization performance.

📊 Dataset Information

Total Samples: 2148

Features: 16 morphological attributes

Target Classes:

0 → Kirmizi_Pistachio

1 → Siirt_Pistachio

Train/Test Split: 80/20 (Stratified)

⚙️ Preprocessing Steps

Removed missing class labels

Cleaned label formatting

Encoded categorical target into numerical values

Applied StandardScaler for feature normalization

Used stratified splitting to preserve class distribution

🧠 Model Architecture

Model: MLPClassifier

Hidden Layers: (32, 16)

Activation Function: ReLU

Max Iterations: 500

Random State: 42

📈 Results

Test Accuracy: 90.9%

Classification Report

Kirmizi Precision: 0.92

Kirmizi Recall: 0.92

Siirt Precision: 0.89

Siirt Recall: 0.90

The confusion matrix shows strong class separation with minimal misclassification.

Total test samples: 430
Total misclassifications: 39

📂 Project Structure
pistachio-species-classification/
│
├── data/
│   └── Pistachio_28_Features_Dataset.xlsx
│
├── models/
│   ├── egitilmis_model.pkl
│   └── veri_olcekleme_araci.pkl
│
├── notebook/
│   └── pistachio_classification.ipynb
│
├── requirements.txt
└── README.md

💾 Saved Model

The trained model and scaler are saved in the models/ directory.

You can load them directly without retraining:

import joblib

model = joblib.load("models/egitilmis_model.pkl")
scaler = joblib.load("models/veri_olcekleme_araci.pkl")

▶️ How to Run

Install required libraries:

pip install -r requirements.txt


Run the notebook
or

Use the saved model for inference

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

💡 Key Insight

Morphological features provide strong discriminative power when combined with neural network models.
The model achieves high accuracy with balanced precision and recall across both classes.
