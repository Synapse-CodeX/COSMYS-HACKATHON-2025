import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to the test data folder')
args = parser.parse_args()
val_dir = Path(args.data_path)


model = load_model("models/best_model.h5")

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    str(val_dir),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

val_generator.reset()
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator)
y_pred = (y_pred_probs > 0.5).astype("int").reshape(-1)

print("✔️ Accuracy:", accuracy_score(y_true, y_pred))
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Female", "Male"]))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
