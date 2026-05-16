## 📂 Dataset Used

This project uses the **Brain Tumor Classification (MRI)** dataset from Kaggle.

---

## ⭐ Why This Dataset Was Chosen

This dataset is widely used for deep learning and medical image classification projects because it is:

- ✅ Popular and trusted among students and researchers
- ✅ Cleanly organized into training and testing folders
- ✅ Ready for CNN model training
- ✅ Contains multiple brain tumor categories
- ✅ Suitable for Flask web deployment projects
- ✅ Well documented with many Kaggle notebooks and examples
- ✅ Beginner-friendly while still useful for advanced AI projects

### 📊 Dataset Statistics

- 99K+ downloads
- 800+ public notebooks
- Frequently used in AI/ML medical imaging projects

---

## 🧠 Dataset Classes

| Class | Description |
|---|---|
| Glioma Tumor | Tumor originating from glial brain tissue |
| Meningioma Tumor | Tumor found in the meninges surrounding the brain |
| Pituitary Tumor | Tumor in the pituitary gland |
| No Tumor | Healthy brain MRI scans |

---

## 📁 Dataset Structure

```text
dataset/
│
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
│
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

---

## 📸 Total Images Used

| Type | Number of Images |
|---|---|
| Training Images | 2870 |
| Testing Images | 394 |
| Total Images | 3264 |

---

## ⚙️ Image Processing

Before training, all MRI images were:

- Resized to **150 × 150**
- Normalized to pixel range **0–1**
- Augmented using:
  - Rotation
  - Zoom
  - Width/height shifting

This improves model generalization and reduces overfitting.

---

## 🚀 How the Dataset Is Used

The dataset is used to train a **Convolutional Neural Network (CNN)** that:

- Accepts MRI brain scans as input
- Extracts image features using convolution layers
- Classifies scans into one of four tumor categories
- Generates prediction confidence scores
- Produces Grad-CAM heatmaps for visualization
