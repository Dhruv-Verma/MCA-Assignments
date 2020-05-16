import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import pickle
from random import randint as randi

data = pickle.load(open('Testing/audio_data.pkl','rb'))

noise_path = 'Dataset/_background_noise_/'
noise_data = []
for j in os.listdir(noise_path):
    noise_data.append(librosa.load(noise_path+'\\'+j, sr=None)[0][:16000])
noise_data = np.array(noise_data)

pickle.dump(noise_data, open('noise_data.pkl', 'wb'))

def norm(x):
    return (x-np.mean(x))/np.std(x) 

# Random augmentation
alpha = 0.1
noisy_data = []
for d, sr, l in data:
    noisy = norm(d) + (alpha * norm(noise_data[randi(0,5)])) 
    noisy_data.append((noisy, sr, l))
noisy_data = np.array(noisy_data)

pickle.dump(noisy_data, open('noisy_data.pkl', 'wb'))
