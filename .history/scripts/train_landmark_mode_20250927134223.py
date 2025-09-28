import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def load_landmark_data(landmark_dir):
    X = []
    y = []
    class_names = sorted(os.listdir(landmark_dir))
    for idx, cls in enumerate(class_names):
        cls_folder = os.path.join(landmark_dir, cls)
        for fname in os.listdir(cls_folder):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(cls_folder, fname))
                X.append(arr)
                y.append(idx)
    return np.array(X), np.array(y), class_names

def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    landmark_dir = os.path.join("dataset", "landmarks")
    X, y, class_names = load_landmark_data(landmark_dir)
    print("Classes:", class_names)
    print("Shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X.shape[1], len(class_names))
    model.summary()

    os.makedirs("saved_models", exist_ok=True)
    checkpoint = callbacks.ModelCheckpoint("saved_models/landmark_model.h5", save_best_only=True, monitor="val_accuracy")
    early = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint, early])

    model.save("saved_models/landmark_final.h5")
    with open("saved_models/landmark_labels.json", "w") as f:
        import json
        json.dump(class_names, f)

    print("Landmark model training done.")

if __name__ == "__main__":
    main()
