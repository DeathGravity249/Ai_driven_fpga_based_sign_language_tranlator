import cv2
import numpy as np
import tensorflow as tf
import os

# Load the model
model = tf.keras.models.load_model("saved_model")  # Make sure this path is correct

# Define class names in the same order used during training
class_names = ['Chai', 'India', 'Lassi', 'Udhar', 'Vishal']

# Initialize camera
cap = cv2.VideoCapture(0)

# Constants
IMG_SIZE = 128

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Define region of interest (ROI) square
    box_size = 224
    center_x, center_y = width // 2, height // 2
    top_left_x = center_x - box_size // 2
    top_left_y = center_y - box_size // 2
    bottom_right_x = center_x + box_size // 2
    bottom_right_y = center_y + box_size // 2

    # Draw green rectangle for ROI
    cv2.rectangle(frame, 
                  (top_left_x, top_left_y), 
                  (bottom_right_x, bottom_right_y), 
                  (0, 255, 0), 2)

    # Extract ROI from frame
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    # Only process if ROI has correct size
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        continue

    # Show ROI to verify what model sees
    cv2.imshow("ROI", roi)

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_normalized = roi_rgb.astype("float32") / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Predict
    prediction = model.predict(roi_expanded, verbose=0)[0]
    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = prediction[pred_index]

    # Show prediction
    label = f"Pred: {pred_class} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Translator", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
