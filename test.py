import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("rock_paper_scissors_model.keras")  # Ensure the model file name matches

# Reverse class mapping
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

# Get the image file path from command-line arguments
filepath = sys.argv[1]

# Prepare the image
img = cv2.imread(filepath)
if img is None:
    print("Error: Unable to load image. Please check the file path.")
    sys.exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
img = cv2.resize(img, (224, 224))  # Resize to match the model's input shape
img = img / 255.0  # Normalize the image (same as during training)

# Predict the move
pred = model.predict(np.array([img]))  # Add batch dimension
move_code = np.argmax(pred[0])  # Get the predicted class index
move_name = mapper(move_code)  # Map the index to the class name

# Output the result
print("Predicted: {}".format(move_name))