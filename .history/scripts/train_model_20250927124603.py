# scripts/train_model.py
import os, json, argparse
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks

def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    # built-in augmentation (optional but useful)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)

    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    data_dir = os.path.join("dataset", "processed")
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "test")  # using test as validation here

    img_size = args.img_size
    batch = args.batch
    seed = 123

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                  image_size=(img_size, img_size),
                                                                  batch_size=batch,
                                                                  seed=seed)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir,
                                                                image_size=(img_size, img_size),
                                                                batch_size=batch,
                                                                shuffle=False)

    class_names = train_ds.class_names
    print("Classes:", class_names)
    num_classes = len(class_names)

    # cache & prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = build_model((img_size, img_size, 3), num_classes)
    model.summary()

    os.makedirs("saved_models", exist_ok=True)
    checkpoint = callbacks.ModelCheckpoint("saved_models/best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
    early = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[checkpoint, early])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    args = p.parse_args()
    main(args)
