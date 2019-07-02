#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:47:37 2019

@author: sleek_eagle
"""

import numpy as np
from os import walk
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from os.path import isfile, join
from common import get_ext_paths

import audioread
import decode

savee_data = "/home/sleek_eagle/Downloads/dev/aac/id00024/5OKm0fUMYS8/"
spectrogram_path = "/home/sleek_eagle/research/emotion_recognition/data/savee/spectograms/"


FFT_WINDOW_SIZE = 25 #in ms
FFT_OVERLAP = 0 #overlap raio of the window

#get spectrogram from wav file
def get_spectrogram(file_path):
    sample_rate, samples = wavfile.read(file_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, 
                                                         sample_rate,
                                                         window  = "hamming",
                                                         nperseg = int(FFT_WINDOW_SIZE/1000 * sample_rate),
                                                         noverlap = FFT_OVERLAP)
    return frequencies, times, spectrogram



def plot_spectrogram(frequencies, times, spectrogram):
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

#data_path : path to wav data files
#spec_path : path spectrograms will be saved
def save_spectrograms(data_path = savee_data,spec_path=spectrogram_path): 
    files = get_ext_paths(data_path,"wav")
    for i,file in enumerate(files):
        frequencies, times, spectrogram = get_spectrogram(file)
        max_time = round(times[-1],2)
        max_freq = int(frequencies[-1])
        filename = file.split("/")[-1][0:-4]
        dirname = file.split("/")[-2]
        filename = dirname + "_" + filename+"_"+str(max_time)+"_"+str(max_freq)
        np.save(spec_path +filename ,spectrogram)




with audioread.audio_open("/home/sleek_eagle/Downloads/dev/aac/id00024/5OKm0fUMYS8/00014.m4a") as f:
    print(f.channels, f.samplerate, f.duration)
    print(f.samples)
 







from __future__ import print_function
import audioread
import sys
import os
import wave
import contextlib


tmp_wav_path = "/home/sleek_eagle/research/emotion_recognition/code/audio_CNN/tmp.wav"
def decode(filename):
    filename = os.path.abspath(os.path.expanduser(filename))
    if not os.path.exists(filename):
        print("File not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with audioread.audio_open(filename) as f:
            print('Input file: %i channels at %i Hz; %.1f seconds.' %
                  (f.channels, f.samplerate, f.duration),
                  file=sys.stderr)
            print('Backend:', str(type(f).__module__).split('.')[1],
                  file=sys.stderr)

            with contextlib.closing(wave.open(tmp_wav_path, 'w')) as of:
                of.setnchannels(f.channels)
                of.setframerate(f.samplerate)
                of.setsampwidth(2)

                for buf in f:
                    of.writeframes(buf)

    except audioread.DecodeError:
        print("File could not be decoded.", file=sys.stderr)
        sys.exit(1)
        
decode("/home/sleek_eagle/Downloads/dev/aac/id00024/5OKm0fUMYS8/00014.m4a")

frequencies, times, spectrogram=get_spectrogram(tmp_wav_path)

