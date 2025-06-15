import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow CPU instruction messages

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Set Dataset Path
IMG_SAVE_PATH = 'image_data'

# Class Labels
CLASS_MAP = {"rock": 0, "paper": 1, "scissors": 2, "none": 3}
NUM_CLASSES = len(CLASS_MAP)

def mapper(val):
    return CLASS_MAP[val]

# Load images from dataset directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        dataset.append([img, directory])

# Prepare Data
data, labels = zip(*dataset)
labels = list(map(mapper, labels))
labels = to_categorical(labels, NUM_CLASSES)

# Define the Model
def get_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Compile and Train Model
model = get_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Convert data and labels to numpy arrays
data = np.array(data) / 255.0  # Normalize pixel values to [0, 1]
labels = np.array(labels)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

# Save the model in the newer .keras format
model.save('rock_paper_scissors_model.keras')