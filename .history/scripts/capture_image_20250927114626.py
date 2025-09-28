import cv2
import os
import time

DATASET_PATH = "../dataset/raw/"
LABEL = ""   # Change label as needed
SAVE_PATH = os.path.join(DATASET_PATH, LABEL)
os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
TOTAL_IMAGES = 200

print(f"Capturing images for label: {LABEL}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Capture", frame)

    if count < TOTAL_IMAGES:
        img_name = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1
        print(f"Saved image {count}/{TOTAL_IMAGES}")
    else:
        print("Finished capturing images.")
        break

    # Add a small delay to avoid capturing frames too fast
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for label: {LABEL}")
