import os
import cv2
import mediapipe as mp
import numpy as np
import json

mp_hands = mp.solutions.hands

def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None
        # take first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        return np.array(coords).flatten()  # shape (21 × 3) → 63 features

def main():
    raw_dir = os.path.join("dataset", "raw")
    out_dir = os.path.join("dataset", "landmarks")
    # structure: landmarks/class_name/*.npy
    os.makedirs(out_dir, exist_ok=True)
    
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    for cls in classes:
        cls_raw = os.path.join(raw_dir, cls)
        cls_out = os.path.join(out_dir, cls)
        os.makedirs(cls_out, exist_ok=True)

        for fname in os.listdir(cls_raw):
            if fname.lower().endswith((".png",".jpg",".jpeg")):
                in_path = os.path.join(cls_raw, fname)
                lm = extract_landmarks(in_path)
                if lm is not None:
                    out_path = os.path.join(cls_out, fname + ".npy")
                    np.save(out_path, lm)
                # else: skip images where no hand found

    print("Landmark extraction done.")

if __name__ == "__main__":
    main()
