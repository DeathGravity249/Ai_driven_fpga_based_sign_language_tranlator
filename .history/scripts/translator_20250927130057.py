# scripts/translator.py
import cv2, json, os, argparse
import numpy as np
import tensorflow as tf

def load_labels(path="saved_models/labels.json"):
    with open(path, "r") as f:
        return json.load(f)

def preprocess_img(img, size):
    import cv2
    img = cv2.resize(img, (size, size))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def main(args):
    model = tf.keras.models.load_model(args.model)
    labels = load_labels()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    side = min(w, h) // 2
    cx, cy = w // 2, h // 2
    x1, y1 = cx - side//2, cy - side//2
    x2, y2 = x1 + side, y1 + side

    last_label = None
    tts = None
    if args.speak:
        try:
            import pyttsx3
            tts = pyttsx3.init()
        except Exception as e:
            print("pyttsx3 not available:", e)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        roi = frame[y1:y2, x1:x2]
        inp = preprocess_img(roi, args.img_size)
        pred = model.predict(inp)
        idx = int(pred.argmax())
        label = labels[idx]

        # overlay
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"Pred: {label}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
        cv2.imshow("Translator", frame)

        if args.speak and label != last_label and tts is not None:
            tts.say(label); tts.runAndWait()
            last_label = label

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="saved_models/best_model.h5")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--speak", action="store_true", help="use text-to-speech for predictions")
    args = p.parse_args()
    main(args)
