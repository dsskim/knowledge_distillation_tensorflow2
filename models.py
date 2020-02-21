import tensorflow as tf
from tensorflow.keras import layers
    
def large_model(input_size, target_cls, dropout=False):
    model = tf.keras.Sequential(name='large_model')
    model.add(layers.Dense(1200, input_shape=(input_size,), activation='relu'))
    if dropout == True:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1200, activation='relu'))
    if dropout == True:
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(target_cls, name='logits'))
    model.add(layers.Activation('softmax', name='softmax'))
    
    model.summary()

    return model

def small_model(input_size, target_cls):
    model = tf.keras.Sequential(name='small_model')
    model.add(layers.Dense(10, input_shape=(input_size,), activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(target_cls, name='logits'))
    model.add(layers.Activation('softmax', name='softmax'))

    model.summary()

    return model