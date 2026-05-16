# 🧠 Brain Tumor Classification Model

This folder contains the trained deep learning model used in the **NeuroScan AI — Brain Tumor Detection System**.

---

## 📌 Model Information

| Property | Value |
|---|---|
| Model Type | Convolutional Neural Network (CNN) |
| Framework | TensorFlow / Keras |
| File Format | `.h5` |
| Input Shape | 150 × 150 × 3 |
| Classes | 4 Brain MRI Categories |
| Output Layer | Softmax |
| Task | Multiclass Image Classification |

---

## 🧬 Classes Predicted

The model predicts one of the following MRI scan categories:

- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

---

## 📂 Dataset Used

The model was trained using the **Brain Tumor Classification (MRI)** dataset from Kaggle.

### Dataset Features

- MRI brain scan images
- Organized train/test folders
- 4 tumor classes
- Widely used for medical imaging projects

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Epochs | 25 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |

---

## 🚀 Model Features

- CNN-based MRI image classification
- Image preprocessing and normalization
- Data augmentation during training
- Probability prediction scores
- Grad-CAM heatmap visualization support
- Flask backend integration

---

## 📁 Model File

```text
model.h5
```

This file stores:

- Model architecture
- Learned weights
- Optimizer state
- Training configuration

---

## 💻 Usage

Load the model in Python using TensorFlow:

```python
from tensorflow.keras.models import load_model

model = load_model("model.h5")
```
