# 🧬 Cancer Image Classification — ITC Challenge

> 🥉 **3rd Place — ITC Challenge: Deep Learning for Precision Oncology**

A deep learning project for **Malignant vs. Benign cancer image classification**, developed for the ITC Challenge.

The solution uses **Transfer Learning, ConvNeXt-Tiny, Stratified 5-Fold Cross-Validation, and model ensembling**, with **ROC-AUC** as the main evaluation metric.

---

## 🎯 Project Overview

The objective is to classify medical images into two classes:

* `0` — Benign
* `1` — Malignant

The dataset contains **10,000 images**, including **7,000 training images and 3,000 test images**.

The model generates a **malignant probability** for each image rather than only a binary label.

---

## 🧠 Methodology

```text
Medical Images
      ↓
Exploratory Data Analysis
      ↓
Image Preprocessing
      ↓
Data Augmentation
      ↓
Stratified 5-Fold Cross-Validation
      ↓
ConvNeXt-Tiny + Transfer Learning
      ↓
ROC-AUC Evaluation
      ↓
5-Fold Ensemble
      ↓
Final Test Predictions
      ↓
submission_final.csv
```

### Data Augmentation

Training images are augmented using:

* Horizontal Flip
* Vertical Flip
* Rotation
* Affine Transformations
* Perspective Transformations
* Color Jitter
* Resize
* Normalization

---

## 🧠 Model

The main architecture is **ConvNeXt-Tiny** with pretrained weights.

Transfer learning is used to adapt the pretrained model to the binary cancer image classification task.

The final classification layer produces a probability for the **Malignant** class.

---

## 🔄 5-Fold Cross-Validation

The training dataset is divided using **Stratified 5-Fold Cross-Validation**.

A separate ConvNeXt-Tiny model is trained for each fold, and the best model from each fold is saved according to validation **ROC-AUC**.

The five models are then combined through probability averaging.

---

## 📊 Evaluation

The main evaluation metric is:

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between malignant and benign images based on predicted probabilities.

---

## 🤝 Ensemble

The final prediction is calculated by averaging the predictions of the five fold-specific models:

```text
Fold 1 ─┐
Fold 2 ─┤
Fold 3 ─┼──► Average Predictions ──► Final Probability
Fold 4 ─┤
Fold 5 ─┘
```

The final predictions are saved in the required CSV format:

```csv
image_id,label
image_1,0.82
image_2,0.14
```

---

## 💻 Technologies

* Python
* PyTorch
* ConvNeXt-Tiny
* timm
* scikit-learn
* OpenCV
* NumPy
* Jupyter Notebook
* Google Colab
* Google Drive
* Kaggle

---

## ☁️ Execution Environment

The current notebook is designed to run in **Google Colab with Google Drive**.

The notebook mounts Google Drive and accesses the dataset from the configured Drive directories.

To reproduce the project:

1. Download the competition dataset.
2. Place it in the expected Google Drive directory.
3. Open `ITC_Challenge.ipynb` in Google Colab.
4. Mount Google Drive.
5. Install the required dependencies.
6. Run the notebook.

---

## 🏆 Competition

**ITC Challenge — Deep Learning for Precision Oncology**

**Result: 🥉 3rd Place**

The challenge focuses on malignant vs. benign medical image classification.

🔗 Kaggle:
https://www.kaggle.com/competitions/cancer-image-classification-itc-challenge/data

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes**.

It is **not a medical diagnostic system** and should not be used for clinical decision-making.

---

## 👩‍💻 Author

**Chaima Mansouri**

Deep Learning • Computer Vision • Medical AI • Artificial Intelligence
