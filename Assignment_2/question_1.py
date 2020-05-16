import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

# os.chdir('C:/Users/dhruv/Desktop/MCA Assignment-2/Dataset/training/one/')
# file_name = 'C:/Users/dhruv/Desktop/MCA Assignment-2/Dataset/training/one/0a7c2a8d_nohash_0.wav'
# file_name = 'sample.wav'
# signal, s_freq = librosa.load(file_name, sr=None) # Sampled at ordiginal sampling rate
# print('duration of the audio: '+ str(len(signal)/s_freq))
# plt.plot(signal)

def dft(x):
    #  Ref: http://www.robots.ox.ac.uk/~sjrob/Teaching/SP/l7.pdf
    x = np.array(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(x):
    #  Ref: http://www.robots.ox.ac.uk/~sjrob/Teaching/SP/l7.pdf
    x = np.array(x, dtype=float)
    N = len(x)
    if N % 2 > 0:
        print("signal length should be even")
    elif N <= 16:   # Optimized by trial
        return dft(x)
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        x_fft = np.concatenate([even + terms[:int(N/2)] * odd,
                               even + terms[int(N/2):] * odd])
        return x_fft
      
def fft_freq(x, s_freq):
    ft = fft(x)
    N= len(ft)
    rfft = np.abs(ft[:N//2])
    freq = np.arange(0, s_freq//2, s_freq/N)
    return (rfft, freq)
    
def spectrogram(x, s_freq, window_size = 256, overlap = 0.5, eps = 1e-14):
    step_size = int(overlap * window_size)
    windows = []
    times = []
    for i in range(0, len(x)-window_size, step_size):
        windows.append(x[i:i+window_size])
        times.append(i/s_freq)
    windows = np.asarray(windows, dtype=float)
    weights = np.hanning(window_size) # Used Hanning Window
    hann_windows = windows * weights
    
    fft_windows = []
    freqs = []
    scale = np.sum(weights**2) * s_freq
    for window in hann_windows:
        ft, freqs = fft_freq(window, s_freq)
        # Power calculation
        ft = np.square(ft)  
        # Scaling
        ft[1:-1] *= (2.0 / scale)
        ft[[0, -1]] /= scale
        # log
        ft = np.log10(ft + eps)
        fft_windows.append(ft)
    fft_windows = np.asarray(fft_windows, dtype=float).T
    freqs = np.asarray(freqs, dtype=float)
    spec = pd.DataFrame(fft_windows, index= freqs, columns=times)
    # ax = sns.heatmap(spec, cmap='viridis')
    # ax.invert_yaxis()
    return spec
    