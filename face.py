import cv2
from deepface import DeepFace
import os

# Step 1: Capture your face image for registration
def capture_face(image_path='registered_face.jpg'):
    cap = cv2.VideoCapture(0)
    print("Press 's' to save your face image.")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(image_path, frame)
            print(f"Face image saved as {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 2: Continuously check if face matches
"""def recognize_face(image_path='registered_face.jpg'):
    if not os.path.exists(image_path):
        print("No registered face found. Please run capture_face() first.")
        return

    cap = cv2.VideoCapture(0)
    print("Recognizing your face...")

    while True:
        ret, frame = cap.read()
        try:
            result = DeepFace.verify(frame, image_path, enforce_detection=False)
            if result["verified"]:
                text = "person is Present"
            else:
                text = "person is Missing"
        except:
            text = "Face not clear"

        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()"""

# Step-by-step run
# First run this to capture your face
# capture_face()

# Then run this to recognize in live feed
# recognize_face()
