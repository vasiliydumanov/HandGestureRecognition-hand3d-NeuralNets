import os
import numpy as np
import keras
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
import coremltools
from utils import input_output_to_float32
import pandas as pd


def prepare_keypoints_data(use_augmented=True, convert_to_categorical=True):
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    keypoints_dir_name = "augmented_keypoints" if use_augmented else "keypoints"
    keypoints_files_dir = os.path.join(parent_dir, f"video_to_keypoints/{keypoints_dir_name}")
    keypoints_files = os.listdir(keypoints_files_dir)
    xs = []
    ys = []
    class_names = []
    for idx, file in enumerate(keypoints_files):
        file_path = os.path.join(keypoints_files_dir, file)
        keypoints = np.load(file_path)
        class_labels = np.full([len(keypoints)], idx, dtype=np.int32)
        xs.append(keypoints)
        ys.append(class_labels)

        file_name = file.rsplit('.', 1)[0]
        class_names.append(file_name)

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    x /= 256.0
    if convert_to_categorical:
        y = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    return x_train, x_test, y_train, y_test, class_names


def create_sequential_model():
    model_depth = 4
    layer_size = 400
    dropout_rate = 0.3

    model = Sequential()
    model.add(
        Dense(layer_size, activation='relu', input_shape=(2 * 21,)))
    model.add(
        Dropout(dropout_rate))
    for _ in range(model_depth - 1):
        model.add(
            Dense(layer_size, activation='relu'))
        model.add(
            Dropout(dropout_rate))
    model.add(
        Dense(5, activation='softmax'))

    return model


def train_sequential(use_augmented=True):
    model = create_sequential_model()

    x_train, x_test, y_train, y_test, _ = prepare_keypoints_data(use_augmented=use_augmented)

    from shutil import rmtree
    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, "sequential")
    if os.path.exists(model_dir):
        rmtree(model_dir)
    os.mkdir(model_dir)
    model_path = os.path.join(model_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    final_path = os.path.join(model_dir, "final.hdf5")

    model.compile(optimizer=Adam(lr=0.001),
                  loss=categorical_crossentropy,
                  metrics=["accuracy"])
    model.fit(x=x_train,
              y=y_train,
              batch_size=32,
              epochs=1000,
              verbose=2,
              callbacks=[
                  ModelCheckpoint(filepath=model_path,
                                  monitor="val_acc",
                                  save_best_only=True,
                                  save_weights_only=True,
                                  verbose=1,
                                  period=10),
                  ReduceLROnPlateau(monitor="val_loss",
                                    factor=0.2,
                                    patience=30,
                                    verbose=1),
                  EarlyStopping(monitor="val_loss",
                                min_delta=0,
                                patience=200,
                                verbose=1,
                                restore_best_weights=True)
              ],
              validation_split=0.2)
    model.save_weights(final_path)
    loss, acc = model.evaluate(x_test, y_test, batch_size=4, verbose=1)
    print("test_loss:", loss, "test_acc", acc)


def convert_sequential_model():
    x_train, _, _, _, class_names = prepare_keypoints_data(use_augmented=False)

    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, "sequential")
    model = create_sequential_model()
    model.load_weights(os.path.join(model_dir, "final.hdf5"))

    coreml_model = coremltools.converters.keras.convert(model=model,
                                                        input_names=["keypoints"],
                                                        output_names=["output"],
                                                        class_labels=class_names)
    coreml_model_path = os.path.join(current_dir, "out/GestureNet.mlmodel")
    spec = coreml_model.get_spec()

    metadata = spec.description.metadata
    metadata.author = "Vasilii Dumanov"
    metadata.shortDescription = "FFNN classifier for prediction of 1-5 numbers based on hand keypoints"

    input_output_to_float32(spec)
    coremltools.utils.save_spec(spec, coreml_model_path)


def show_sequential_confusion_matrix():
    x_train, _, y_train, _, class_names = prepare_keypoints_data(use_augmented=True)

    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, "sequential")
    model = create_sequential_model()
    model.load_weights(os.path.join(model_dir, "final.hdf5"))
    preds = model.predict(x_train)
    preds = np.argmax(preds, axis=1)
    y = np.argmax(y_train, axis=1)

    cm = confusion_matrix(y, preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)


def inspect_coreml_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "GestureNet.mlmodel")

    spec = coremltools.utils.load_spec(model_path)
    print(spec)


def main():
    # train_sequential(use_augmented=True)
    convert_sequential_model()


if __name__ == "__main__":
    main()
