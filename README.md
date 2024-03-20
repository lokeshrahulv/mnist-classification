# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
## Neural Network Model

![Screenshot 2024-03-20 140701](https://github.com/lokeshrahulv/mnist-classification/assets/118423842/0814d493-d525-4b6e-bc95-aa1cf8a41262)

## DESIGN STEPS

## STEP 1:
Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.
## STEP 2:
Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.
## STEP 3:
Compile the model with categorical cross-entropy loss function and the Adam optimizer.
## STEP 4:
Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.
## STEP 5:
Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.

## PROGRAM
### Name:LOKESH RAHUL V V
### Register Number:212222100024
```python

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()

model.add(layers.Input(shape=(28,28,1)))

model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

print('Lokesh Rahul V V 212222100024')
metrics.head()

print('Lokesh Rahul V V 212222100024')
metrics[['accuracy','val_accuracy']].plot()

print('Lokesh Rahul V V 212222100024')
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print('Lokesh Rahul V V 212222100024')
print(confusion_matrix(y_test,x_test_predictions))

print('Lokesh Rahul V V 212222100024')
print(classification_report(y_test,x_test_predictions))

img = image.load_img('imgthree.jpg')

type(img)

img_tensor1 = tf.convert_to_tensor(np.asarray(img))
img_28_gray1 = tf.image.resize(img_tensor1,(28,28))
img_28_gray1 = tf.image.rgb_to_grayscale(img_28_gray1)
img_28_gray= 255.0-img_28_gray1
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction1 = np.argmax(
    model.predict(img_28_gray_inverted_scaled1.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction1)

print('Lokesh Rahul V V 212222100024')
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img1 = image.load_img('imgfive.jpg')

img_tensor1 = tf.convert_to_tensor(np.asarray(img1))
img_28_gray1 = tf.image.resize(img_tensor1,(28,28))
img_28_gray1 = tf.image.rgb_to_grayscale(img_28_gray1)
img_28_gray_inverted1 = 255.0-img_28_gray1
img_28_gray_inverted_scaled1 = img_28_gray_inverted1.numpy()/255.0

x_single_prediction1 = np.argmax(
    model.predict(img_28_gray_inverted_scaled1.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction1)
```



## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-03-20 135305](https://github.com/lokeshrahulv/mnist-classification/assets/118423842/8888879b-313e-4292-b2d7-dcc0277d9344)
![Screenshot 2024-03-20 135322](https://github.com/lokeshrahulv/mnist-classification/assets/118423842/dd7906e3-e5c9-4261-adf5-184cc8091828)
![Screenshot 2024-03-20 135331](https://github.com/lokeshrahulv/mnist-classification/assets/118423842/7abc55e6-489d-4a92-b86f-461551a262af)

### Classification Report
![Screenshot 2024-03-20 135340](https://github.com/lokeshrahulv/mnist-classification/assets/118423842/534ca0ee-59df-4eba-b0e1-662417d150c2)

### Confusion Matrix
![Screenshot 2024-03-20 135347](https://github.com/lokeshrahulv/mnist-classification/assets/118423842/d9e5a5ae-55b7-4fd2-bb77-e466c816d608)

### New Sample Data Prediction
![Screenshot 2024-03-20 135357](https://github.com/lokeshrahulv/mnist-classification/assets/118423842/13618e8c-1319-4421-81f5-7d11ef733283)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
