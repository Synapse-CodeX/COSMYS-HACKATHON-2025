import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess

train_dir = "/kaggle/input/cosmys-hackathon-5/Comys_Hackathon5/Task_B/train"
val_dir = "/kaggle/input/cosmys-hackathon-5/Comys_Hackathon5/Task_B/val"

train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator()
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

def build_mesonet(input_shape):
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(64, activation='relu')
    ])
    return model


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

def mesonet_preprocess(x):
    return x / 255.0

train_dual = DualInputGenerator(train_gen, xcep_preprocess, mesonet_preprocess)
val_dual = DualInputGenerator(val_gen, xcep_preprocess, mesonet_preprocess)


input_x = Input(shape=(224, 224, 3))
xcep_base = Xception(weights='imagenet', include_top=False, input_tensor=input_x)
x = GlobalAveragePooling2D()(xcep_base.output)
x = Dense(64, activation='relu')(x)
input_m = Input(shape=(224, 224, 3))
meso_model = build_mesonet((224, 224, 3))
m = meso_model(input_m)

merged = concatenate([x, m])
z = Dense(128, activation='relu')(merged)
z = Dropout(0.5)(z)
output = Dense(train_gen.num_classes, activation='softmax')(z)

model = Model(inputs=[input_x, input_m], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model_taskB.keras", monitor='val_accuracy', save_best_only=True)

history = model.fit(
    train_dual,
    validation_data=val_dual,
    epochs=25,
    callbacks=[early_stop, checkpoint]
)

model.save("dual_model_taskB_final.h5")
