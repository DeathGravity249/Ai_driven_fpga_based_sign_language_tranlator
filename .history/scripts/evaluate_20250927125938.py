# scripts/evaluate.py
import os, json, argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main(args):
    model_path = args.model
    labels_path = "saved_models/labels.json"
    with open(labels_path, "r") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path)
    test_dir = os.path.join("dataset", "processed", "test")
    img_size = args.img_size
    batch = args.batch

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, image_size=(img_size,img_size),
                                                                   batch_size=batch, shuffle=False)
    # get y_true
    y_true = np.concatenate([y.numpy() for x,y in test_ds], axis=0)
    # predictions
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    # plot confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Saved outputs to outputs/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="saved_models/best_model.h5")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    main(args)
