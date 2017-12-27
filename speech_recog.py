
# coding: utf-8

# In[2]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.
  Args:
    filename: Path to the .wav file to load.
  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


# In[3]:


ip_sound= load_wav_file("/home/richa/Documents/mydocs/kaggle/train/audio/bed/00176480_nohash_0.wav")


# In[4]:


def prepare_background_data():
    """Searches a folder for background noise audio, and loads it into memory.
    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.
    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.
    Returns:
      List of raw PCM-encoded audio samples of background noise.
    Raises:
      Exception: If files aren't found in the folder.
    """
    background_data = []
    #background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    background_dir = "/home/richa/Documents/mydocs/kaggle/train/audio/_background_noise_"
    if not os.path.exists(background_dir):
      return background_data
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(background_dir,'*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        background_data.append(wav_data)
      if not background_data:
        raise Exception('No background wav files were found in ' + search_path)
      return background_data


# In[5]:


noise=prepare_background_data()


# In[6]:


import scipy.io.wavfile
import scipy.signal
# Read the .wav file
#sample_rate, data = scipy.io.wavfile.read('/home/richa/Documents/mydocs/kaggle/train/audio/bed/00176480_nohash_0.wav')
sample_rate, data = scipy.io.wavfile.read("/home/richa/Documents/mydocs/kaggle/train/audio/sheila/004ae714_nohash_0.wav")
# Spectrogram of .wav file
sample_freq, segment_time, spec_data = scipy.signal.spectrogram(data, sample_rate)  
# Note sample_rate and sampling frequency values are same but theoretically they are different measures


# In[7]:


import matplotlib.pyplot as plt
plt.pcolormesh(segment_time, sample_freq, spec_data )
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()  


# In[9]:


from scipy.io import wavfile
from scipy import signal
import numpy as np

sample_rate, audio = wavfile.read("/home/richa/Documents/mydocs/kaggle/train/audio/sheila/004ae714_nohash_0.wav")

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, spec

sample_freq, segment_time, spec_data = log_specgram(audio,sample_rate)

import matplotlib.pyplot as plt
plt.pcolormesh(segment_time, sample_freq, spec_data)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show() 

