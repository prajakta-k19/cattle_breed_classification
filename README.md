# 🐄🐃 AI-Based Cow & Buffalo Breed Classification Using Custom CNN

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Research paper implementation — *AI-Based Cow & Buffalo Breed Classification Using Custom CNN*  
> KIIT Deemed to be University, Bhubaneswar, India

---

## 📌 Overview

An end-to-end deep learning system that classifies **64 Indian cow and buffalo breeds** from images using a custom-built Convolutional Neural Network (CNN). The system achieves **92.5% accuracy** on the validation set and is deployed as a Streamlit web application with Google OAuth login.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **92.5%** |
| Weighted Precision | 0.928 |
| Weighted Recall | 0.931 |
| Weighted F1-Score | 0.929 |
| Specificity | 0.941 |
| No. of Breeds | 64 |
| Dataset Size | ~8,700 images (after preprocessing) |

---

## 🧠 Model Architecture

Custom CNN with **5 convolutional blocks** followed by dense layers:

```
Input (128×128×3)
  → Conv2D(32) + MaxPool
  → Conv2D(64) + MaxPool
  → Conv2D(128) + MaxPool
  → Conv2D(256) + MaxPool
  → Conv2D(512) + MaxPool
  → Dense(512) + Dropout(0.5)
  → Dense(256) + Dropout(0.5)
  → Dense(64, softmax)
```

Outperforms VGG16, ResNet50, DenseNet121, MobileNetV2, and EfficientNetB0 on this dataset.

---

## 📂 Repository Structure

```
animal_breed_classification/
├── AI_Breed_Classification_Complete.ipynb  # Full pipeline: preprocessing → training → export
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
├── models/                                 # Saved model files
│   ├── animal_classifier_savedmodel/       # TF SavedModel (used by app.py)
│   └── model.json                          # Class index → breed name mapping
└── .devcontainer/                          # GitHub Codespaces config
```

---

## ⚙️ Pipeline (`AI_Breed_Classification_Complete.ipynb`)

| Phase | Description |
|-------|-------------|
| **1 — Setup** | Mount Drive, install dependencies, define unified paths |
| **2 — Dataset Organization** | Flatten `species/breed` folders, 80/10/10 train/val/test split |
| **3 — Preprocessing** | Duplicate removal (imagehash), blur detection (OpenCV), corrupt image cleanup |
| **4 — Augmentation** | Flip, rotate, zoom, brightness & contrast variation (3× effective dataset size) |
| **5 — Model Training** | 5-block Custom CNN, 100 epochs, Adam optimizer, early stopping |
| **6 — Evaluation** | Confusion matrix, classification report, Grad-CAM visualizations |
| **7 — Export** | `.keras` model, TF SavedModel, `model.json` class mapping |

---

## 🚀 Running the App

```bash
# Clone the repository
git clone https://github.com/swaraj3092/animal_breed_classification.git
cd animal_breed_classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app supports:
- 📁 Image upload (JPG, PNG, JPEG)
- 📸 Live camera capture
- 🔐 Google OAuth + email/password login
- 📊 Top-3 breed predictions with confidence scores
- 🕓 Prediction history per session

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **Preprocessing:** OpenCV, Pillow, imagehash
- **App:** Streamlit, SQLite
- **Auth:** Google OAuth 2.0
- **Data:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

---

## 📊 Dataset

- ~9,200 raw images → **8,700 after preprocessing**
- **64 breeds** (e.g., Gir, Sahiwal, Murrah, Jafarabadi, Punganur, Toda, Hariana, etc.)
- Sources: ICAR-NDRI, DAHD, Kaggle Cattle Breeds, Dairy DigiD, Google Open Images, field photographs from Odisha
- Augmented to ~3× original size for training robustness

---

## 👥 Authors

| Name | Institution |
|------|------------|
| Amiya Ranjan Panda | KIIT University |
| **Swaraj Kumar Behera** | KIIT University |
| Prajakta Kuila | KIIT University |
| Vidya Mohanty | Aryan Institute of Engineering and Technology |
| Subhashree Mishra | KIIT University |
| Manoj Kumar Mishra | KIIT University |

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you find this project useful, give it a star!
