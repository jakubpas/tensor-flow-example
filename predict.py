#!/usr/bin/python3
import os
import string
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys


# It's important to include all custom functions related to deserialization
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    stripped_html = tf.strings.regex_replace(stripped_html, '<p>', ' ')
    stripped_html = tf.strings.regex_replace(stripped_html, '</p>', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def main():
    filename = sys.argv[1]
    text = open(filename, 'r').read()
    model = tf.keras.models.load_model('trained_model')
    # model.summary()

    print(1 if model.predict([text])[0][0] > 0.5 else 0)


if __name__ == '__main__':
    main()
