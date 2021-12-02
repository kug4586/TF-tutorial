import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import extract
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract = True,
        cache_dir = '.', cache_subdir = 'data'
    )

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
#print('Commadns : ', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
# print('Number of total examples : ', num_samples)
# print('Number of examples per label : ', len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
# print('Example file tensor : ', filenames[0])

train_files = filenames[:6400]
val_files = filenames[6400 : 6400 + 800]
test_files = filenames[-800:]
# print('Training set size : ', len(train_files))
# print('Validation set size : ', len(val_files))
# print('Test set size : ', len(test_files))

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

