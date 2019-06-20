#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:27:04 2019

@author: sleek_eagle
"""
import numpy as np
from common import get_ext_paths
from create_spectrogram import plot_spectrogram
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle
import mini_xception
import keras
import importlib
from random import randint



#lenght of the time segment for classification (in seconds)
WINDOW = 1.0
#ratio of overlap between two adjecent window. ration taken with respect to the length of window
OVERLAP = 0.4
#read from data stored as spectograms
data_path  = "/home/sleek_eagle/research/emotion_recognition/data/savee/spectograms"
#input shape for the NN
INPUT_SHAPE = (48,48,1)


class_names = np.array(['a', 'd', 'f', 'h', 'n', 'sa', 'su'])

def get_spectrogram_windows(spectrogram,max_time):
    samples_per_window = int(spectrogram.shape[1]/max_time*WINDOW)
    samples_per_progress = int(spectrogram.shape[1]/max_time*(1-OVERLAP))
    spec_list = []
    for i in range(0,spectrogram.shape[1],samples_per_progress):
        if(i > spectrogram.shape[1]-samples_per_window):
            break
        spec = spectrogram[:,i:i+samples_per_window]
        spec = resize(spec,INPUT_SHAPE)
        log_spec = log_spectrogram(spec,1e-10) 
        spec_list.append(log_spec)
    return spec_list

def log_spectrogram(spectrogram,c):
    return np.log(spectrogram+c)

def normalize_spectrogram(spectrogram):
    np.mean(spectrogram,axis=1)
      
def get_spectrogram(max_freq,max_time,spectrogram):
    times = []
    num = spectrogram.shape[1]
    inc=max_time/num
    for i in range(1,num+1):
        times.append(i*inc)
        
    frequencies = []
    num = spectrogram.shape[0]
    inc=max_freq/num
    for i in range(1,num+1):
        frequencies.append(i*inc)
        
    plot_spectrogram(np.array(frequencies),np.array(times),spectrogram)
    

def plot_spectrogram_hist(spectrogram):   
    spec = spectrogram.flatten()
    spec = pd.Series(spec)
    
    spec.plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
    plt.grid(axis='y', alpha=0.75)
    
 
def get_data_split(ratios = [0.6,0.8]):
    files = get_ext_paths(data_path,"npy")
    shuffle(files)
    indices = (np.array(ratios)*len(files)).astype(int)
    train = files[0:indices[0]]
    vali = files[indices[0]:indices[1]]
    test = files[indices[1]:-1]
    return train,vali,test

        
def to_onehot(num):
    out = np.empty([0,len(class_names)])
    for x in np.nditer(num):
        onehot = np.zeros(len(class_names))
        onehot[int(x)] = 1
        out = np.append(out,[onehot],axis = 0)
    return out

def get_emo_num(emotion):
    num = -1
    try:
        num = np.where(class_names == emotion)[0][0]
        num = to_onehot(num)
    except:
        print("cannot find this emotion name in the data...")
    return num

#prepare data for instance based classification
def get_data(files):
    data = []
    labels = []
    for file in files:
        spectrogram  = np.load(file)
        split = file.split("/")[-1].split("_")
        max_time = float(split[2])
        max_freq = int(split[-1].split('.')[0])
        emotion = get_emo_num(split[1][0:-2])
        spectrograms = get_spectrogram_windows(spectrogram,max_time)
        for spec in spectrograms:
            data.append(spec)
            labels.append(emotion)
    data = np.array(data)
    labels = np.array(labels)
    labels = np.reshape(labels,(labels.shape[0],labels.shape[-1]))
    return data,labels

#get 'num' random samples from data where label is emotion (choosen from class_names)
def get_random_samples(labels,data,emotion,num):
    emotion_num = np.where(class_names == emotion)[0][0]
    class_nums = np.argmax(labels,axis=1) 
    class_indices = np.where(class_nums == emotion_num)[0]
    np.random.shuffle(class_indices)
    class_indices = class_indices[0:num]
    class_data = np.take(data,class_indices,axis=0)
    return class_data
    
#get a batch from data of size and neutral_ratio of neutral data in it
#after this data is randomly shuffled    
def get_batch(data,labels,size,neutral_ratio):    
    neutral_size = int(size*neutral_ratio)
    class_size = size - neutral_size
    neutral_data = get_random_samples(labels,data,'n',neutral_size)  
    select_class = class_names[randint(0,len(class_names))]
    onehot_class = get_emo_num(select_class)
    class_data = get_random_samples(labels,data,select_class,class_size)
    data = np.append(neutral_data,class_data,axis=0)
    np.random.shuffle(data)
    labels = np.repeat(onehot_class,size,axis=0)
    return data,labels

#for bathc wise learning (with temporal pooling)
d,l = get_batch(data,labels,100,0.6)

    
train,vali,test = get_data_split()
train_data,train_labels = get_data(train)
vali_data,vali_labels = get_data(vali)
test_data,test_labels = get_data(test)



batch_size = 32
epochs = 100
importlib.reload(mini_xception)
model = mini_xception.get_xception()


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.00001)

model.fit(x = train_data,y=train_labels,batch_size=batch_size,epochs=epochs,validation_data=[vali_data,vali_labels])

model.evaluate(x=test_data,y=test_labels)



        




