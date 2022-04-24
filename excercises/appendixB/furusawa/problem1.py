import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import mnist

"""
Problem 1
=========

Go to the Keras MNIST example. Perform a classification task.
Note that how many epochs the training takes, and how well it generalizes.
Perform the classification on a smaller training set, how does learning rate change, how does generalization change.
Vary other elements:
  - try a different optimizer than adam
  - try a different learning rate
  - try a different(deeper) architecture
  - try wider hidden layers.
Does it learn faster? Does it generalize better?
"""

def mnist_classification(validation_split=0, optimizer="adam", depth=0, epochs=5, width=32, batch_size=128):
  # loading data
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # reshape data
  train_images = train_images.reshape((60000, 28 * 28))
  train_images = train_images.astype('float32')/255

  test_images = test_images.reshape((10000, 28 * 28))
  test_images = test_images.astype('float32')/255

  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)

  # modelの作成
  model = Sequential()

  model.add(Dense(width, activation="relu", input_shape=(28 * 28,)))
  for _ in range(depth):
    model.add(Dense(width, activation="relu"))
  model.add(Dense(10, activation="softmax"))

  model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

  # トレーニング
  history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

  loss_and_metrics = model.evaluate(test_images, test_labels, batch_size=batch_size)

  print(list(zip(model.metrics_names, loss_and_metrics)))



if __name__ == "__main__":
  print("Default Parameters")
  mnist_classification()
  print("------------------")
  print("Less Samples: validation_split=0.9")
  mnist_classification(validation_split=0.9)
  print("------------------")
  print("Another Optimizer: optimizer='sgd'")
  mnist_classification(optimizer="sgd")
  print("------------------")
  print("Deeper layers: depth=4")
  mnist_classification(depth=4)
  print("------------------")
  print("Wide hidden layers: width=256")
  mnist_classification(width=256)

"""
Expected Output
===============

Default Parameters
Epoch 1/5
469/469 [==============================] - 1s 934us/step - loss: 0.5153 - accuracy: 0.8615
Epoch 2/5
469/469 [==============================] - 0s 928us/step - loss: 0.2539 - accuracy: 0.9288
Epoch 3/5
469/469 [==============================] - 0s 934us/step - loss: 0.2097 - accuracy: 0.9410
Epoch 4/5
469/469 [==============================] - 0s 931us/step - loss: 0.1807 - accuracy: 0.9488
Epoch 5/5
469/469 [==============================] - 0s 927us/step - loss: 0.1579 - accuracy: 0.9553
79/79 [==============================] - 0s 576us/step - loss: 0.1561 - accuracy: 0.9554
[('loss', 0.1561281532049179), ('accuracy', 0.9553999900817871)]
------------------
Less Samples: validation_split=0.9
Epoch 1/5
47/47 [==============================] - 1s 10ms/step - loss: 1.4970 - accuracy: 0.5823 - val_loss: 0.8884 - val_accuracy: 0.8044
Epoch 2/5
47/47 [==============================] - 0s 6ms/step - loss: 0.6239 - accuracy: 0.8608 - val_loss: 0.5399 - val_accuracy: 0.8624
Epoch 3/5
47/47 [==============================] - 0s 7ms/step - loss: 0.4320 - accuracy: 0.8916 - val_loss: 0.4458 - val_accuracy: 0.8794
Epoch 4/5
47/47 [==============================] - 0s 6ms/step - loss: 0.3603 - accuracy: 0.9080 - val_loss: 0.4021 - val_accuracy: 0.8858
Epoch 5/5
47/47 [==============================] - 0s 6ms/step - loss: 0.3175 - accuracy: 0.9147 - val_loss: 0.3733 - val_accuracy: 0.8951
79/79 [==============================] - 0s 617us/step - loss: 0.3577 - accuracy: 0.9023
[('loss', 0.35769250988960266), ('accuracy', 0.9023000001907349)]
------------------
Another Optimizer: optimizer='sgd'
Epoch 1/5
469/469 [==============================] - 0s 804us/step - loss: 1.3422 - accuracy: 0.6684
Epoch 2/5
469/469 [==============================] - 0s 788us/step - loss: 0.6039 - accuracy: 0.8533
Epoch 3/5
469/469 [==============================] - 0s 790us/step - loss: 0.4652 - accuracy: 0.8783
Epoch 4/5
469/469 [==============================] - 0s 798us/step - loss: 0.4092 - accuracy: 0.8883
Epoch 5/5
469/469 [==============================] - 0s 792us/step - loss: 0.3774 - accuracy: 0.8950
79/79 [==============================] - 0s 616us/step - loss: 0.3493 - accuracy: 0.9033
[('loss', 0.34929025173187256), ('accuracy', 0.9032999873161316)]
------------------
Deeper layers: depth=4
Epoch 1/5
469/469 [==============================] - 1s 1ms/step - loss: 0.5511 - accuracy: 0.8236
Epoch 2/5
469/469 [==============================] - 1s 1ms/step - loss: 0.2077 - accuracy: 0.9388
Epoch 3/5
469/469 [==============================] - 1s 1ms/step - loss: 0.1611 - accuracy: 0.9519
Epoch 4/5
469/469 [==============================] - 1s 1ms/step - loss: 0.1354 - accuracy: 0.9594
Epoch 5/5
469/469 [==============================] - 1s 1ms/step - loss: 0.1223 - accuracy: 0.9628
79/79 [==============================] - 0s 664us/step - loss: 0.1427 - accuracy: 0.9591
[('loss', 0.14268070459365845), ('accuracy', 0.9591000080108643)]
------------------
Wide hidden layers: width=256
Epoch 1/5
469/469 [==============================] - 1s 2ms/step - loss: 0.3075 - accuracy: 0.9144
Epoch 2/5
469/469 [==============================] - 1s 2ms/step - loss: 0.1326 - accuracy: 0.9614
Epoch 3/5
469/469 [==============================] - 1s 2ms/step - loss: 0.0912 - accuracy: 0.9737
Epoch 4/5
469/469 [==============================] - 1s 2ms/step - loss: 0.0678 - accuracy: 0.9801
Epoch 5/5
469/469 [==============================] - 1s 2ms/step - loss: 0.0536 - accuracy: 0.9839
79/79 [==============================] - 0s 1ms/step - loss: 0.0740 - accuracy: 0.9759
[('loss', 0.07401996850967407), ('accuracy', 0.9758999943733215)]
"""

