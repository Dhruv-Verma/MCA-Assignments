import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import pickle

os.chdir('C:/Users/dhruv/Desktop/MCA Assignment-2/')
path = 'Dataset/validation/'
print(os.listdir(path))
labels = {'zero':0,
          'one':1,
          'two':2,
          'three':3,
          'four':4,
          'five':5,
          'six':6,
          'seven':7,
          'eight':8,
          'nine':9}

data = []
for i in os.listdir(path):
    print(i)
    label = labels[i]
    for j in os.listdir(path+i):
        d = {}
        signal, s_freq = librosa.load(path+i+'\\'+j, sr=None) # Sampled at ordiginal sampling rate
        data.append((signal, s_freq, label))
    
data = np.array(data)

# Length Correction
i=0
for d, sr, l in data:
    if(len(d)!=16000):
        zero=np.zeros(16000-len(d))
        data[i][0] = np.concatenate((d, zero))
    i+=1

pickle.dump(data, open('audio_data.pkl', 'wb'))
data = pickle.load(open('Noisy\\Testing\\noisy_data.pkl','rb'))

#------------------------------------------------
# Spectrograms
from question_1 import spectrogram

specs = []
labels = data[:,2]
for d, sr, l in data:
    s = np.array(spectrogram(d, sr)).flatten()
    specs.append(s)
specs_vector = np.array(specs)
labels = np.array(labels)
pickle.dump(specs_vector, open('specs_vector.pkl','wb'))
pickle.dump(labels, open('labels.pkl','wb'))
       
#----------------------------------------------
# MFCC 
from question_2 import mfcc

mfccs = []
for d, sr, l in data:
    s = np.array(mfcc(d,sr,320,post_process=False)).flatten()
    mfccs.append(s)
mfccs_vector = np.array(mfccs)
pickle.dump(mfccs_vector, open('mfccs_vector.pkl','wb'))

#----------------------------------------------
# MFCC post-processed
from question_2 import mfcc

mfccs_post = []
for d, sr, l in data:
    s = np.array(mfcc(d,sr,320,post_process=True)).flatten()
    mfccs_post.append(s)
mfccs_vector_post = np.array(mfccs_post)
pickle.dump(mfccs_vector_post, open('mfccs_vector_post.pkl','wb'))

#---------------------------------------------
