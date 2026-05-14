# AI-Based Cow & Buffalo Breed Classification Using Custom CNN

> **Published Research** — 2026 IEEE International Conference  
> Co-authored by Prajakta Kuila, Swaraj Kumar Behera, Amiya Ranjan Panda et al., KIIT Deemed to be University

[![IEEE](https://img.shields.io/badge/IEEE-Published-blue)](https://ieeexplore.ieee.org/document/11496187)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.5%25-brightgreen)]()
[![Breeds](https://img.shields.io/badge/Breeds-64-orange)]()
[![Python](https://img.shields.io/badge/Python-3.x-yellow)]()

---

## Overview

This project presents a deep learning system for automatically classifying **64 Indian cow and buffalo breeds** from images using a custom-designed Convolutional Neural Network (CNN). The model achieves **92.5% overall accuracy** and outperforms established architectures including ResNet50, VGG16, DenseNet121, MobileNetV2, and EfficientNetB0.

Accurate breed identification is critical for precision livestock management — enabling selective breeding, health monitoring, and dairy yield prediction. Conventional manual inspection is labor-intensive and error-prone; this system automates that process end-to-end.

---

## Key Results

| Metric | Value |
|---|---|
| Overall Accuracy | **92.5%** |
| Weighted Precision | 0.928 |
| Weighted Recall | 0.931 |
| Weighted F1-Score | **0.929** |
| Specificity | 0.941 |

### Comparison with Pretrained Architectures

| Model | Accuracy | F1-Score |
|---|---|---|
| VGG16 | 0.852 | 0.84 |
| ResNet50 | 0.887 | 0.88 |
| DenseNet121 | 0.895 | 0.89 |
| MobileNetV2 | 0.901 | 0.90 |
| EfficientNetB0 | 0.912 | 0.91 |
| **Custom CNN (Ours)** | **0.925** | **0.929** |

---

## Dataset

- **9,200 high-quality images** across 64 Indian cattle breeds (e.g., Gir, Sahiwal, Murrah, Jafarabadi)
- Sources: ICAR-NDRI, DAHD, Kaggle Cattle Breeds dataset, Dairy DigiD, and field-level farm photographs from Odisha
- **3x augmentation** applied — random rotation, flipping, brightness/saturation adjustments, zoom, shear, and Gaussian noise injection
- Images collected across varied lighting, pose, and background conditions for real-world generalization

---

## Model Architecture

Custom 5-block CNN built from scratch, optimized for livestock visual features:

- **Conv Blocks**: 5 blocks with 32 → 64 → 128 → 256 → 512 filters, ReLU activation, 2×2 max pooling
- **Regularization**: Dropout (rate = 0.5) after each dense layer to prevent overfitting
- **Output Layer**: 64 neurons with softmax activation (one per breed class)
- **Training**: Adam optimizer, categorical cross-entropy loss, learning rate 0.001, 100 epochs, batch size 32, early stopping

---

## Methodology

1. **Data Collection** — Curated from government databases, open datasets, and rural farm photography
2. **EDA** — Breed distribution analysis, image quality checks, visual feature study
3. **Preprocessing** — Resizing to 224×224, normalization, noise reduction, histogram equalization, center cropping
4. **Augmentation** — 3x dataset expansion via rotation, flipping, brightness, zoom, shear, Gaussian noise
5. **Model Training** — Custom CNN trained on GPU with early stopping and LR reduction callbacks
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Specificity, Confusion Matrix
7. **Interpretability** — Grad-CAM heatmaps validating model focus on breed-discriminative features

---

## Interpretability — Grad-CAM

Grad-CAM visualizations confirmed the model correctly focuses on breed-specific features:
- Horn curvature and orientation
- Coat patterns and markings
- Facial structure and ear shape

Misclassifications were primarily limited to visually similar or rare breeds (e.g., Punganur and Toda) due to limited training samples.

---

## Tech Stack

- **Language**: Python 3.x
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: Pillow (PIL)
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib
- **App Interface**: Streamlit (`app.py`)

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/prajakta-k19/cattle_breed_classification
cd cattle_breed_classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Applications

- Precision livestock management and selective breeding
- Automated dairy farm monitoring
- Veterinary health tracking and yield prediction
- Integration with IoT devices for real-time edge inference

---

## Publication

> Amiya Ranjan Panda, Swaraj Kumar Behera, **Prajakta Kuila**, Vidya Mohanty, Subhashree Mishra, Manoj Kumar Mishra.  
> *"AI-Based Cow & Buffalo Breed Classification Using Custom CNN"*  
> 2026 IEEE International Conference — [View on IEEE Xplore](https://ieeexplore.ieee.org/document/11496187)

---

## Authors

- **Prajakta Kuila** — KIIT Deemed to be University
- **Swaraj Kumar Behera** — KIIT Deemed to be University
- Amiya Ranjan Panda, Vidya Mohanty, Subhashree Mishra, Manoj Kumar Mishra
