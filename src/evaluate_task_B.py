import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
from tensorflow.keras.utils import Sequence

class DualInputGenerator(tf.keras.utils.Sequence):
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

def mesonet_preprocess(x):
    return x / 255.0

val_dir = "/kaggle/input/cosmys-hackathon5/Comys_Hackathon5/Task_B/val"
model_path = "models/best_model_taskB.h5"
output_file = "task_B_evaluation_report.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

val_datagen = ImageDataGenerator()
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_dual = DualInputGenerator(val_gen, xcep_preprocess, mesonet_preprocess)

model = tf.keras.models.load_model(model_path)

preds = model.predict(val_dual, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)

print("========== [Evaluation Report: Task B] ==========\n")
print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%\n")
print("📌 Classification Report:")
print(report)
print("📌 Confusion Matrix:")
print(conf_matrix)

with open(output_file, "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))


