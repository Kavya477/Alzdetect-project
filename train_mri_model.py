import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "mri_images")
MODEL_PATH = os.path.join(BASE_DIR, "mri_mobilenet.h5")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10


def build_mobilenet_model(num_classes=1):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    if num_classes == 1:
        predictions = Dense(1, activation="sigmoid")(x)
    else:
        predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy" if num_classes == 1 else "categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    model = build_mobilenet_model(num_classes=1)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
    )

    model.save(MODEL_PATH)
    print(f"MRI MobileNet model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()