import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

val_dir = "/kaggle/input/cosmys-hackathon5/Comys_Hackathon5/Task_A/val"
model_path = "models/best_model.h5"
output_report = "task_A_evaluation_report.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = load_model(model_path)

val_datagen = ImageDataGenerator()
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

def mesonet_preprocess(x):
    return x / 255.0

class DualInputGenerator(Sequence):
    def __init__(self, base_gen, preprocess1, preprocess2):
        self.base_gen = base_gen
        self.preprocess1 = preprocess1
        self.preprocess2 = preprocess2

    def __len__(self):
        return len(self.base_gen)

    def __getitem__(self, idx):
        images, labels = self.base_gen[idx]
        input1 = self.preprocess1(images.copy())
        input2 = self.preprocess2(images.copy())
        return (input1, input2), labels

val_dual = DualInputGenerator(val_gen, xcep_preprocess, mesonet_preprocess)

pred_probs = model.predict(val_dual, verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_gen.classes

acc = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, target_names=val_gen.class_indices.keys())
cm = confusion_matrix(true_labels, pred_labels)

print(f"Accuracy: {acc * 100:.2f}%")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", cm)

with open(output_report, "w") as f:
    f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
