This uses CNNs to classify vocal emotions. 
We first convert audio clips to spectrograms and use them to train a CNN classifier. 
Important files :
create_spectrogram.py : code to create and save spectrograms 
common.py : file access
classify_audio.py : secondary pr-processing of spectrograms and train/test CNN classifier
alexnet.py, mini_xception.py : two CNNs we can use as models
Although experiments are only done with mini_xception, alexnet should also work similarly. 

Audioread 2.1.8 is used to read audio files other than .wav. 
For experiments, SAVEE dataset (which contains acted emotional speach as well as video) is used. 
Dataset can be found at http://kahlan.eps.surrey.ac.uk/savee/

some results of training can be found in the results folder.
