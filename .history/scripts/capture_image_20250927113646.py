import cv2
import os

# Path where images will be saved
DATASET_PATH = "../dataset/raw/"
LABEL = "C"   # <-- Change this letter each time you capture for a different class
SAVE_PATH = os.path.join(DATASET_PATH, LABEL)

# Create folder if not exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0
TOTAL_IMAGES = 200  # Number of images per letter

print(f"Capturing images for letter: {LABEL}")

while True:
    ret, frame = cap.read()
    print(f"Frame read status: {ret}")  # Add this line

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    cv2.imshow("Capture", frame)

    # Save images as before

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {LABEL}")
