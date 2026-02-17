# deep-learning-lego-image-classification
Clasificación de piezas LEGO mediante CNN con análisis comparativo de técnicas de preprocesamiento usando TensorFlow y OpenCV.

# 🧠 CNN LEGO Classification – Computer Vision Project

Deep Learning project focused on image classification of LEGO pieces using Convolutional Neural Networks (CNN) and advanced image preprocessing techniques.

## 📌 Project Overview

This project implements and evaluates different preprocessing pipelines and CNN architectures for image classification using the **B200C LEGO Classification Dataset** (Kaggle).

The main objective is to analyze how image enhancement techniques affect model performance and generalization.

---

## 🗂 Dataset

- **Dataset:** B200C LEGO Classification Dataset
- **Classes used:** 4 selected LEGO piece categories
- **Images per class:** 4000
- **Image size:** 64x64 (resized to 32x32 for experiments)
- **Format:** RGB (.jpg)
- **Balanced dataset**

---

## ⚙️ Preprocessing Techniques Evaluated

Two preprocessing pipelines were implemented and compared:

### 🔹 Method 1 – Blur + Canny Edge Detection
- Normalization
- Gaussian Blur (3x3)
- Grayscale conversion
- Canny edge detection
- Edge fusion with original image
- Resize to 32x32

### 🔹 Method 2 – Gaussian + Otsu + Contours
- Grayscale conversion
- Gaussian Blur (5x5)
- Otsu thresholding
- Contour detection
- Largest contour drawn over original image
- Resize to 32x32

---

## 🏗 CNN Architecture

The implemented CNN model consists of:

- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling (2x2)
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling (2x2)
- Dropout (0.5)
- Flatten
- Dense (400 neurons, ReLU)
- Dense (4 neurons, Softmax)

Optimizer: Adam  
Loss Function: Categorical Crossentropy  
Metric: Accuracy  

---

## 📊 Experimental Results

Different experiments were performed varying:

- Number of classes
- Number of images
- Image size (32x32 / 64x64)
- Epochs
- Preprocessing technique

### 🔥 Best Balanced Model
- Image size: 32x32
- Epochs: 13
- Preprocessing: Gaussian + Otsu + Contours
- Training Accuracy: ~86%
- Validation Accuracy: ~85%
- Reduced overfitting compared to baseline

---

## 📈 Key Findings

- Preprocessing techniques significantly affect model generalization.
- Edge-based enhancement improves feature extraction but may increase training time.
- Excessive epochs can lead to overfitting.
- Balanced dataset contributes to stable validation accuracy.

---

## 🚀 How to Run

```bash
python main.py -p "path_to_dataset"

