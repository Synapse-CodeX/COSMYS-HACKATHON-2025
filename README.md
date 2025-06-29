# COSMYS Hackathon 5 🚀  
## 🧠 Dual-Model Fusion for Gender Classification and Face Recognition  

This repo contains solutions for both **Task A** (Gender Classification) and **Task B** (Multi-Class Face Recognition with Distortions) using **deep learning** with **dual model fusion**.

---

## 📁 Folder Structure

```
.
├── train_Task_A.py
├── evaluate_Task_A.py
├── train_Task_B.py
├── evaluate_Task_B.py
├── models/
│   ├── best_model_task_A.h5
│   └── best_model_task_B.h5
├── task_A_evaluation_report.txt
├── task_B_evaluation_report.txt
└── README.md
```

---

## ✅ Task A - Gender Classification

### 🔍 Problem
Binary gender classification on a dataset of face images. The model must be robust to lighting, pose, and distortion.

### 🧠 Approach
A **dual-input fusion model** combining:
- `Xception` (pretrained on ImageNet) for global features.
- A custom `MesoNet-like` CNN for local facial patterns.

### ⚙️ Training
```bash
python train_Task_A.py
```

### 🧪 Evaluation
```bash
python evaluate_Task_A.py
```

### 📊 Metrics
- **Accuracy**: ~92.65%
- `task_A_evaluation_report.txt` contains:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

## ✅ Task B - Multi-Class Face Recognition

### 🔍 Problem
Multi-class classification across **N classes** (one per individual), with distortions in the validation set.

### 🧠 Approach
A dual-branch model combining:
- `EfficientNetB0` (ImageNet weights) for high-level embeddings.
- A custom lightweight CNN for robustness to distortion.

### ⚙️ Training
```bash
python train_Task_B.py
```

### 🧪 Evaluation
```bash
python evaluate_Task_B.py
```

### 📊 Metrics
- **Accuracy**: Varies with class count, ~85–90%
- `task_B_evaluation_report.txt` includes:
  - Top-1 Accuracy
  - Classification Report
  - Confusion Matrix

---

## 📦 Model Weights
After training, best models are saved automatically to:
```
models/best_model_task_A.h5
models/best_model_task_B.h5
```

---

## 🔧 Requirements

```bash
pip install tensorflow numpy scikit-learn
```

---


## 📌 Notes
- Both models were trained and evaluated on **Kaggle GPU runtime**.
- Directory structure must be preserved as:
```
Task_A/
  └── train/
       ├── male/
       └── female/
  └── val/
       ├── male/
       └── female/

Task_B/
  └── train/
       ├── 0/
       ├── 1/
       └── ...
  └── val/
       ├── 0/
       ├── 1/
       └── ...
```

---

## 🏁 Submission Guidelines

✔️ Metrics clearly mentioned  
✔️ Clean and documented code  
✔️ Trained model weights provided  
✔️ Test script and evaluation report included

---
