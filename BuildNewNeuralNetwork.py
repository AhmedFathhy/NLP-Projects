# build first neural network
import tensorflow as tf
import numpy as np
# define the mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f'x_train shape: {x_train.shape}')
# print(f'y_train shape: {y_train.shape}')
# print(f'x_test shape: {x_test.shape}')
# print(f'y_test shape: {y_test.shape}')
# ---------------------The above lines are for debugging purposes---------------------
# review the data of the first image
# print(x_train[0])
# build the model
model = tf.keras.models.Sequential()  # create a sequential model
# add the first layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # input layer (28x28) -> (784) flattened in one dimension
# add the hidden layer
model.add(
    tf.keras.layers.Dense(128, activation=tf.nn.relu))  # hidden layer (784) -> (128) with relu activation function
# dropout layer
model.add(tf.keras.layers.Dropout(
    0.2))  # dropout layer with 20% of the neurons randomly being dropped out to avoid overfitting
# add the output layer
model.add(
    tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer (128) -> (10) with softmax activation function
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train the model
model.fit(x_train, y_train, epochs=10)  # train the model for 3 epochs (3 times) you can change the epochs to any number you want to train the model for more epochs
# predict the model
# print(model.predict(x_test))  # predict the model on the test data

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)  # evaluate the model on the test data
print(f'loss: {loss}')  # print the loss of the model
print(f'accuracy: {accuracy}')  # print the accuracy
# now save the model
model.save('my_model.model')  # save the model
# you may also save the model to a HDF5 file
# model.save('my_model.h5')  # save the model to a HDF5 file
# you may also save the model to a TF Lite file
# model.save('my_model.tflite')  # save the model to a TF Lite file
# you may load the model for further use by loading it from the file system
new_model = tf.keras.models.load_model('my_model.model')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
