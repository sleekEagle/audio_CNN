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
from common import get_ext_paths


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

    
 
files = get_ext_paths(savee_data,"wav")
for i,file in enumerate(files):
    frequencies, times, spectrogram = get_spectrogram(file)
    max_time = round(times[-1],2)
    max_freq = int(frequencies[-1])
    filename = file.split("/")[-1][0:-4]
    dirname = file.split("/")[-2]
    filename = dirname + "_" + filename+"_"+str(max_time)+"_"+str(max_freq)
    np.save(spectrogram_path +filename ,spectrogram)


   