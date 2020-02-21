import collections
import numpy as np
import tensorflow as tf

from models import large_model, small_model

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255

    train_images = train_images.reshape((len(train_images), -1))
    test_images = test_images.reshape((len(test_images), -1))

    train_labels = tf.keras.utils.to_categorical(train_labels.astype('float32'))
    test_labels = tf.keras.utils.to_categorical(test_labels.astype('float32'))

    model = small_model(train_images.shape[-1], 10)

    opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='student', histogram_freq=1)

    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10000, batch_size=100, callbacks=[tensorboard_callback])

