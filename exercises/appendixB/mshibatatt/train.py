import os
import argparse
import numpy as np
from tensorflow import keras
from keras import layers
import mlflow
from tensorflow.python.client import device_lib

# https://keras.io/examples/vision/mnist_convnet/
def main(batch_size = 128, epochs = 15, optimizer = "adam", learning_rate = 0.001,
         seed = None):
    num_classes = 18
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    print("x_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    mlflow.keras.autolog()

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate = learning_rate)
    elif optimizer == "SDG":
        opt = keras.optimizers.SDG(learning_rate = learning_rate)
    else:
        raise(NotImplementedError("optimizer should be adam or SDG"))

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(X_test, y_test, verbose=0)

    mlflow.log_metric(key = "loss", value = score[0])
    mlflow.log_metric(key = "accuracy", value = score[1])
    mlflow.log_param(key = "batch_size", value = batch_size)
    mlflow.log_param(key = "epochs", value = epochs)
    mlflow.log_param(key = "optimizer", value = optimizer)
    mlflow.log_param(key = "learning_rate", value = learning_rate)
    mlflow.log_param(key = "device", value = device_lib.list_local_devices()[-1])
    mlflow.log_param(key = "seed", value = seed)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--b",
        type=int,
        default=128,
        help="batch_size",
    )
    parser.add_argument(
        "-e", "--e",
        type=int,
        default=15,
        help="epochs",
    )
    parser.add_argument(
        "-lr", "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "-opt", "--opt",
        type=str,
        default="adam",
        help="optimizer: adam or SGD",
    )
    parser.add_argument(
        "-seed", "--seed",
        type=int,
        default=None,
        help="seed number default does not set",
    )
    args = parser.parse_args()
    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)
    main(args.b, args.e, args.opt, args.lr, args.seed)
