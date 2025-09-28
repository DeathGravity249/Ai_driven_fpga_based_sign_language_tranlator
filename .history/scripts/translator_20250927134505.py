import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json

# Load model & labels
model = tf.keras.models.load_model("saved_models/landmark_model.h5")
with open("saved_models/landmark_labels.json", "r") as f:
    class_names = json.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        coords = []
        for lm_point in lm.landmark:
            coords.extend([lm_point.x, lm_point.y, lm_point.z])
        coords = np.array(coords).reshape(1, -1)  # shape (1, 63)
        pred = model.predict(coords)[0]
        idx = np.argmax(pred)
        label = class_names[idx]
        confidence = pred[idx]

        cv2.putText(frame, f"{label} {confidence*100:.1f}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        # No hand detected
        cv2.putText(frame, "No Hand", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Sign Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
