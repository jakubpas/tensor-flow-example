#!/usr/bin/python3
import re
import os
import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

epochs = 30  # How mny training rounds
dictionary_size = 250
batch_size = 32  # How many texts to take at once to train and validation
seed = 42  # Seed for randomization of input sets
dropout = 0.2  # What rate use to zeroing to prevent overwriting
validation_split = 0.3  # What part of data to use for validation


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<p>', ' ')
    stripped_html = tf.strings.regex_replace(stripped_html, '<br />', ' ')
    stripped_html = tf.strings.regex_replace(stripped_html, '</p>', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def main():
    raw_val_ds = tf.keras.preprocessing. \
        text_dataset_from_directory('data/train', batch_size=batch_size, validation_split=validation_split,
                                    subset='validation', seed=seed)
    raw_train_ds = tf.keras.preprocessing. \
        text_dataset_from_directory('data/train', batch_size=batch_size, validation_split=validation_split,
                                    subset='training', seed=seed)
    vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=dictionary_size,
                                        output_mode='int', output_sequence_length=dictionary_size)

    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))  # Create index of words

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        layers.Embedding(input_dim=dictionary_size + 1, output_dim=16),  # Turns integers into vectors of fixed size
        layers.Dropout(dropout),  # Add randomly 0.2 to input to prevent overwriting
        layers.GlobalAveragePooling1D(),  # Takes average among all time steps.
        layers.Dropout(dropout),  # Add randomly 0.2 to input to prevent overwriting
        layers.Dense(1)])  # One dimensional dense output layer

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.summary()

    export_model = tf.keras.Sequential([vectorize_layer, model, layers.Activation('sigmoid')])

    export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False),
                         optimizer="adam", metrics=['accuracy'])
    export_model.predict(['test'], verbose=0)
    export_model.save('trained_model')


if __name__ == '__main__':
    main()
