import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import cv2
import numpy as np
from random import choice
from tensorflow.keras.models import load_model

# Reverse class mapping
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

# Load the model
model = load_model("rock_paper_scissors_model.keras")  # Ensure the model file name matches

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

frame_height, frame_width, _ = frame.shape
print(f"Frame dimensions: {frame_width}x{frame_height}")

# Define rectangle coordinates
user_rect = (50, 50, 300, 300)  # (x1, y1, x2, y2) - Adjusted for 640x480 frame
computer_rect = (350, 50, 600, 300)  # (x1, y1, x2, y2) - Adjusted for 640x480 frame

# Ensure the rectangles fit within the frame
if (computer_rect[2] > frame_width or computer_rect[3] > frame_height):
    print("Error: Computer rectangle is outside the frame dimensions.")
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    print(f"Computer rectangle: {computer_rect}")
    exit()

prev_move = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Rectangle for user to play
        cv2.rectangle(frame, (user_rect[0], user_rect[1]), (user_rect[2], user_rect[3]), (255, 255, 255), 2)
        # Rectangle for computer to play
        cv2.rectangle(frame, (computer_rect[0], computer_rect[1]), (computer_rect[2], computer_rect[3]), (255, 255, 255), 2)

        # Extract the region of image within the user rectangle
        roi = frame[user_rect[1]:user_rect[3], user_rect[0]:user_rect[2]]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to match model's input shape
        img = img / 255.0  # Normalize the image

        # Predict the move made
        pred = model.predict(np.array([img]))  # Add batch dimension
        move_code = np.argmax(pred[0])
        user_move_name = mapper(move_code)

        # Predict the winner (human vs computer)
        if prev_move != user_move_name:
            if user_move_name != "none":
                computer_move_name = choice(['rock', 'paper', 'scissors'])
                winner = calculate_winner(user_move_name, computer_move_name)
            else:
                computer_move_name = "none"
                winner = "Waiting..."
        prev_move = user_move_name

        # Display the information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + user_move_name,
                    (10, 30), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + computer_move_name,
                    (350, 30), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Winner: " + winner,
                    (200, 450), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

        # Display computer's move icon
        if computer_move_name != "none":
            icon_path = f"images/{computer_move_name}.png"
            if os.path.exists(icon_path):
                icon = cv2.imread(icon_path)
                icon = cv2.resize(icon, (computer_rect[2] - computer_rect[0], computer_rect[3] - computer_rect[1]))  # Resize icon to fit the rectangle
                frame[computer_rect[1]:computer_rect[3], computer_rect[0]:computer_rect[2]] = icon
            else:
                print(f"Icon not found: {icon_path}")

        # Display the frame
        cv2.imshow("Rock Paper Scissors", frame)

        # Break the loop if 'q' or 'Esc' is pressed
        k = cv2.waitKey(30)
        print(f"Key pressed: {k}")  # Debugging: Print the key code
        if k == ord('q') or k == 27:  # Close on 'q' or 'Esc'
            break
finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()