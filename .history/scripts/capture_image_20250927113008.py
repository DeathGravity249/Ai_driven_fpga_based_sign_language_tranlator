import cv2
import os

# Path where images will be saved
DATASET_PATH = "../dataset/raw/"
LABEL = "C"   # <-- change this when capturing different letters
SAVE_PATH = os.path.join(DATASET_PATH, LABEL)

# Create folder if not exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0
TOTAL_IMAGES = 200  # number of images per class

print(f"Capturing images for letter: {LABEL}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame
    cv2.imshow("Capture", frame)

    # Save images
    if count < TOTAL_IMAGES:
        img_name = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1
    else:
        break

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {LABEL}")
