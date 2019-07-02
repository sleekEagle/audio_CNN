#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:50:47 2019

@author: sleek_eagle
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:55:30 2019

@author: sleek-eagle
"""
#from __future__ import print_function

import numpy as np
from os import walk
import os
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from os.path import isfile, join
from common import get_ext_paths
import pandas as pd



#for formats which are not wav
import audioread
import sys
import wave
import contextlib


NUM_CLASSES = 6
savee_data = "/home/sleek_eagle/research/emotion_recognition/data/savee/AudioData/"
spectrogram_path = "/home/sleek_eagle/research/emotion_recognition/data/savee/spectograms/"


FFT_WINDOW_SIZE = 25 #in ms
FFT_OVERLAP = 0 #overlap raio of the window

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

def log_spectrogram(spectrogram,c):
    return np.log(spectrogram+c)

def normalize_spectrogram(spectrogram):
    return  (spectrogram - np.mean(spectrogram))/np.std(spectrogram)


def minmax_spectrogram(spectrogram):
    return  (spectrogram - np.min(spectrogram))/(np.max(spectrogram) - np.min(spectrogram))

#plot a histogram from spectrogram
def plot_spectrogram_hist(spectrogram):   
    spec = spectrogram.flatten()
    spec = pd.Series(spec)
    
    spec.plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
    plt.grid(axis='y', alpha=0.75)


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
        
 
path = "/home/sleek_eagle/Downloads/dev/aac/id00012/2DLq_Kkc1r8/00016.m4a"  
def get_filename_data(path):
    split = path.split("/")     
    person = split[-3]
    file = split[-1][0:-4]
    return file,person

'''

#save spectrogram for voxceleb speaker ID data
voxceleb_data = "/home/sleek_eagle/Downloads/aac/"
spectrogram_path = "/home/sleek_eagle/research/emotion_recognition/data/voxceleb_audio/spectrograms/test/"
files = get_ext_paths(voxceleb_data,"m4a")
l=len(files)
for i,file in enumerate(files):
    decode(file)
    frequencies, times, spectrogram = get_spectrogram(tmp_wav_path)
    max_time = round(times[-1],2)
    max_freq = int(frequencies[-1])
    file_name,person = get_filename_data(file)
    #create direcotry if not exist
    
    directory = spectrogram_path + person + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_file = directory + file_name + "_"+str(max_time)+"_"+str(max_freq)
    np.save(save_file ,spectrogram)
    print(str(i/l*100) + "% is done")
    
  

#save spectrograms for savee data set
files = get_ext_paths(savee_data,"wav")
for i,file in enumerate(files):
    frequencies, times, spectrogram = get_spectrogram(file)
    max_time = round(times[-1],2)
    max_freq = int(frequencies[-1])
    filename = file.split("/")[-1][0:-4]
    dirname = file.split("/")[-2]
    filename = dirname + "_" + filename+"_"+str(max_time)+"_"+str(max_freq)
    np.save(spectrogram_path +filename ,spectrogram)
    
'''



   