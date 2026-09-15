# 🧬 Cancer Image Classification — ITC Challenge

> 🥉 **3rd Place — ITC Challenge: Deep Learning for Precision Oncology**

A deep learning project for **Malignant vs. Benign cancer image classification**, developed for the **ITC Challenge** on Kaggle.

The solution combines **Transfer Learning**, **ConvNeXt-Tiny**, **Stratified 5-Fold Cross-Validation**, and **model ensembling** to produce robust probability predictions evaluated using **ROC-AUC**.

---

## 🏆 Competition Result

| Rank                 | Result                              |
| -------------------- | ----------------------------------- |
| 🥉 **3rd Place**     | ITC Challenge                       |
| 🎯 Task              | Malignant vs. Benign Classification |
| 📊 Evaluation Metric | ROC-AUC                             |
| 🧠 Main Architecture | ConvNeXt-Tiny                       |
| 🔄 Validation        | Stratified 5-Fold Cross-Validation  |
| 🤝 Final Prediction  | 5-Fold Ensemble                     |

🔗 **Kaggle Competition:**
https://www.kaggle.com/competitions/cancer-image-classification-itc-challenge/data

---

## 📌 Project Overview

Cancer diagnosis from medical images is a challenging computer vision problem where reliable classification can support clinical decision-making.

The goal of this project was to build a deep learning model capable of distinguishing between:

* 🔴 **Malignant (M)**
* 🟢 **Benign (B)**

The challenge dataset contains **10,000 curated medical images**.

Instead of predicting only a hard class, the competition requires the model to output a **probability between 0 and 1 representing the likelihood that an image is Malignant**.

This makes the problem particularly suitable for evaluating the model using **ROC-AUC**.

---

## 🎯 Objective

The main objective was to develop a robust binary image classifier that:

1. Learns discriminative visual features from medical images.
2. Generalizes well to unseen images.
3. Produces reliable malignant probabilities.
4. Handles the variability of medical imaging data.
5. Achieves strong ROC-AUC performance on the private leaderboard.

The competition also evaluates the **quality of the notebook**, including:

* Exploratory Data Analysis
* Code clarity
* Methodology
* Overall presentation

Therefore, the project focuses not only on model performance but also on creating a clear and reproducible machine learning workflow.

---

# 🧠 Methodology

The complete pipeline can be summarized as:

```text
                    Medical Images
                          │
                          ▼
                ┌───────────────────┐
                │ Exploratory Data  │
                │     Analysis      │
                └─────────┬─────────┘
                          │
                          ▼
                Image Preprocessing
                          │
                          ▼
                Data Augmentation
                          │
                          ▼
              Stratified 5-Fold CV
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
         Fold 1         Fold 2       Fold 3 ... Fold 5
            │             │             │
            ▼             ▼             ▼
       ConvNeXt-Tiny  ConvNeXt-Tiny  ConvNeXt-Tiny
            │             │             │
            └─────────────┼─────────────┘
                          │
                          ▼
                  Probability Ensemble
                          │
                          ▼
                    Final Prediction
                          │
                          ▼
                      Kaggle CSV
```

---

# 🔍 1. Exploratory Data Analysis

Before training the model, the dataset was explored to understand its structure and characteristics.

The analysis includes:

* Dataset structure inspection
* Label distribution
* Image visualization
* Image dimensions
* Missing values
* Duplicate analysis
* Class distribution analysis

Understanding the dataset before training is important, especially in medical image classification where class imbalance and image variability can affect model performance.

---

# 🖼️ 2. Image Preprocessing

Medical images were transformed into a format suitable for deep learning.

The preprocessing pipeline includes:

* Loading image files
* Resizing images to the required input resolution
* Converting images into tensors
* Normalizing pixel values
* Applying training-time transformations

The preprocessing pipeline is designed to provide consistent input to the neural network while preserving relevant visual information.

---

# 🔄 3. Data Augmentation

Data augmentation was used during training to improve the model's ability to generalize to unseen images.

The objective is to expose the model to slightly different versions of the training images while maintaining their original class.

Examples of augmentation strategies include:

* Geometric transformations
* Image flipping
* Rotation
* Other training-time image transformations

Augmentation helps reduce overfitting and improves robustness when the available medical image distribution is limited.

---

# 🧠 4. Model Architecture — ConvNeXt-Tiny

The main backbone used in this project is **ConvNeXt-Tiny**.

ConvNeXt is a modern convolutional architecture that incorporates several design ideas inspired by recent vision architectures while retaining the advantages of convolutional neural networks.

The model was initialized using **pretrained weights**, allowing the network to start from features learned from a large-scale image dataset rather than learning all visual representations from scratch.

This approach is known as:

> **Transfer Learning**

The final classification layer was adapted for the binary cancer classification task.

### Why ConvNeXt-Tiny?

ConvNeXt-Tiny was selected because it provides a strong balance between:

* Representation capacity
* Computational efficiency
* Transfer-learning performance
* Training stability

The architecture is also practical for medical image classification where the available training data may not be sufficient to train a large model entirely from scratch.

---

# 🔄 5. Stratified 5-Fold Cross-Validation

Instead of relying on a single train/validation split, the training data was divided using:

**Stratified 5-Fold Cross-Validation**

```text
Dataset
   │
   ├── Fold 1 → Validation
   │          → Training
   │
   ├── Fold 2 → Validation
   │          → Training
   │
   ├── Fold 3 → Validation
   │          → Training
   │
   ├── Fold 4 → Validation
   │          → Training
   │
   └── Fold 5 → Validation
              → Training
```

The **stratified** strategy preserves the proportion of the two target classes across the folds.

This provides a more reliable estimate of model performance and reduces dependence on one particular validation split.

---

# 🎯 6. Binary Classification

The task is a binary classification problem:

| Class | Meaning   |
| ----- | --------- |
| `0`   | Benign    |
| `1`   | Malignant |

The model produces a continuous probability rather than simply returning a class.

For example:

```text
Prediction = 0.92
```

means the model assigns approximately **92% probability to the Malignant class**.

This probability-based prediction is particularly important because the competition evaluates the model using **ROC-AUC**.

---

# 📊 7. Evaluation — ROC-AUC

The primary evaluation metric is:

## ROC-AUC

The Receiver Operating Characteristic — Area Under the Curve measures how well the model separates malignant and benign cases across different classification thresholds.

Unlike simple accuracy, ROC-AUC evaluates the quality of the model's ranking of positive and negative examples.

Conceptually:

```text
Higher AUC
     │
     ▼
Better separation
     │
     ▼
Malignant probabilities
        vs.
Benign probabilities
```

The competition uses the **Private Leaderboard** as the main ranking component, with the final ranking based primarily on the private evaluation set.

---

# 🤝 8. 5-Fold Ensemble

After training one model for each fold, predictions from the five models were combined.

```text
Fold 1 Model ──┐
Fold 2 Model ──┤
Fold 3 Model ──┼──► Average Predictions ──► Final Probability
Fold 4 Model ──┤
Fold 5 Model ──┘
```

The final prediction is obtained by combining the probability predictions generated by the five fold-specific models.

This ensemble strategy helps reduce the variance associated with an individual model and provides more stable predictions.

---

# 🚀 Training Pipeline

The complete training workflow is:

```text
1. Load dataset
       ↓
2. Explore data
       ↓
3. Analyze class distribution
       ↓
4. Preprocess images
       ↓
5. Apply augmentation
       ↓
6. Create Stratified 5-Fold splits
       ↓
7. Initialize pretrained ConvNeXt-Tiny
       ↓
8. Fine-tune model
       ↓
9. Evaluate using ROC-AUC
       ↓
10. Save best model for each fold
       ↓
11. Generate test predictions
       ↓
12. Ensemble predictions
       ↓
13. Create Kaggle submission
```

---

# 📄 Submission Format

The competition requires a CSV file containing:

```csv
image_id,label
d6b382c028e6,0.95
dec58b144499,0.12
1f3d808d5c8e,0.88
```

Where:

* `image_id` identifies the image.
* `label` is the predicted probability of **Malignant**.

Therefore:

```text
0.00 → very low malignant probability
0.50 → uncertain / intermediate probability
1.00 → very high malignant probability
```

---

# ⚙️ Technologies

The project uses the following technologies:

* 🐍 Python
* 🔥 PyTorch
* 🧠 ConvNeXt-Tiny
* 🖼️ Computer Vision
* 🔄 Transfer Learning
* 📊 ROC-AUC
* 🔁 Stratified K-Fold Cross-Validation
* 🤝 Ensemble Learning
* 📓 Jupyter Notebook
* 🏆 Kaggle

---

# 📚 Reproducibility

To reproduce the project:

1. Download the competition dataset from Kaggle.
2. Place the dataset in the appropriate data directory.
3. Install the required dependencies.
4. Run the notebook or training scripts.
5. Train the five cross-validation folds.
6. Generate predictions for the test set.
7. Ensemble the predictions.
8. Generate the final submission CSV.

The original competition dataset is not redistributed in this repository.

---

# 🏆 Competition

**ITC Challenge: Deep Learning for Precision Oncology — Malignant vs. Benign Classification**

The challenge focuses on applying deep learning to medical image classification, with the goal of distinguishing malignant from benign cases.

The competition evaluates both:

* **Model Performance**
* **Notebook Quality**

The final leaderboard emphasizes performance on the **Private Leaderboard**, helping reduce overfitting to the public evaluation set.

🔗 **Kaggle:**
https://www.kaggle.com/competitions/cancer-image-classification-itc-challenge/data

---

# ⚠️ Disclaimer

This project is developed for **educational and research purposes** as part of a machine learning competition.

It is **not a medical diagnostic system** and should not be used to make clinical decisions.

The predictions generated by the model should not replace professional medical evaluation.

---

# 👩‍💻 Author

**Chaima Mansouri**

Deep Learning • Computer Vision • Medical AI • Artificial Intelligence

