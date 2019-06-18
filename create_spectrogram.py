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
import numpy as np
from os import walk
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from os.path import isfile, join


NUM_CLASSES = 6
savee_data = "/home/sleek_eagle/research/emotion_recognition/data/savee/AudioData/"
spectrogram_path = "/home/sleek_eagle/research/emotion_recognition/data/savee/spectograms/"


FFT_WINDOW_SIZE = 25 #in ms
FFT_OVERLAP = 0.6 #overlap raio of the window

def get_spectrogram(file_path):
    sample_rate, samples = wavfile.read(file_path)
    frequencies, times, spectrogram = signal.spectrogram(samples, 
                                                         sample_rate,
                                                         window  = "hamming",
                                                         nperseg = int(FFT_WINDOW_SIZE/1000 * sample_rate),
                                                         noverlap = 0)
    return frequencies, times, spectrogram


def get_wav_paths(data):
    files = []
    for (dirpath, dirnames, filenames) in walk(data):
        #files.extend(filenames)
        files.append([dirpath,filenames])
        
    audio_files = []
    for item in files:
        dirname = item[0]
        filenames = item[1]
        for filename in filenames:
            if (filename[-3:] != 'wav'):
                continue
            audio_files.append(dirname + "/" + filename)
    
    return audio_files


def plot_spectrogram(frequencies, times, spectrogram):
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    
 
times_file = open(spectrogram_path + "times.txt","w+")
freq_file = open(spectrogram_path + "freq.txt","w+")

files = get_wav_paths(savee_data)
for i,file in enumerate(files):
    frequencies, times, spectrogram = get_spectrogram(file)
    filename = file.split("/")[-1][0:-4]
    dirname = file.split("/")[-2]
    filename = dirname + "_" + filename
    np.save(spectrogram_path +filename ,spectrogram)
    times_file.write(filename + ',' + ','.join([str(x) for x in list(times)]) + "\n")
    freq_file.write(filename + ',' +','.join([str(x) for x in list(frequencies)]) + "\n")
    
times_file.close() 
freq_file.close() 

   