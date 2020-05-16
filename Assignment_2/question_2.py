import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from scipy.fftpack import dct

# os.chdir('C:/Users/dhruv/Desktop/MCA Assignment-2/Dataset/training/one/')
# file_name = 'C:/Users/dhruv/Desktop/MCA Assignment-2/Dataset/training/one/0a7c2a8d_nohash_0.wav'
# file_name = 'sample.wav'
# signal, s_freq = librosa.load(file_name, sr=None) # Sampled at ordiginal sampling rate


# print('duration of the audio: '+ str(len(signal)/s_freq))
# plt.plot(signal)

def stft(x, s_freq, window_size = 256, overlap = 0.5, eps = 1e-14):
    step_size = int(overlap * window_size)
    windows = []
    times = []
    for i in range(0, len(x)-window_size, step_size):
        windows.append(x[i:i+window_size])
        times.append(i/s_freq)
    windows = np.asarray(windows, dtype=float)
    weights = np.hanning(window_size) # Used Hanning Window
    hann_windows = windows * weights
    
    # Fourier Transform
    fft_windows = []
    fft_windows = np.fft.fft(hann_windows, axis=-1)
    N = fft_windows.shape[1]
    
    # Coeff for positive frequencies
    fft_windows = fft_windows[:,:N//2]
    
    # Spectral Power 
    fft_windows = np.square(np.abs(fft_windows)) 
    
    # Scaling (to accomodate Hanning window)
    scale = np.sum(weights**2) * s_freq
    fft_windows[:,1:-1] *= (2.0 / scale)
    fft_windows[:,[0, -1]] /= scale
    
    # Taking loge
    # fft_windows = np.log(fft_windows + eps).T
    
    # Frequencies 
    fft_freq = np.fft.fftfreq(N, 1/s_freq)
    fft_freq = fft_freq[:N//2]
    
    # spec[freq][time] = power
    spec = pd.DataFrame(fft_windows.T, index= fft_freq, columns=times)
    return spec
  
def mel(f):
    return 2595. * np.log10(1. + f / 700.)

def melinv(m):
    return 700. * (np.power(10., m / 2595.) - 1.)

def mel_filters(min_hz, max_hz, s_freq, fft_size, nfilters=26):
    # Ref: http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank
    min_mel = mel(min_hz)
    max_mel = mel(max_hz)
    mels = np.linspace(min_mel, max_mel, nfilters+2)
    freqs = melinv(mels)
    f_indices = np.floor((fft_size)/s_freq * freqs).astype(int)
    
    # Create filter bank
    f_bank = np.zeros([nfilters, fft_size//2]) 
    for j in range(nfilters):
        # Line equation from left to centre
        for x in range(int(f_indices[j]), int(f_indices[j+1])):
            f_bank[j,x] = (x - f_indices[j]) / (f_indices[j+1] - f_indices[j])
        # Line equation from centre to right
        for x in range(int(f_indices[j+1]), int(f_indices[j+2])):
            f_bank[j,x] = (f_indices[j+2] - x) / (f_indices[j+2] - f_indices[j+1])
    return f_bank

def mfcc(signal, s_freq, window_size = 256, overlap = 0.50, n_ceps = 13, post_process=False, eps = 1e-14):
    # Ref: https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
    # Pre-emphasis
    pre_emphasis = 0.97
    signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    pow_frames = np.array(stft(signal, s_freq, window_size)).T
    filter_bank = mel_filters(min_hz=0, max_hz=(s_freq//2), s_freq=s_freq, fft_size=window_size).T
    cc = np.matmul(pow_frames, filter_bank)     # Cepstral coefficients
    # Convert to dB
    cc = np.log10(cc+ eps)
    # Applying DCT
    mfcc = dct(cc, type=2, axis=1, norm='ortho')
    mfcc = mfcc[:, 1: (n_ceps+1)]
    if(post_process):
        # Sinusoidal liftering
        x = np.arange(n_ceps)    
        cep_lifter = 22
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * x / cep_lifter)
        mfcc *= lift
        # Mean normalize
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc.T
   
    
# mfcc1 = mfcc(signal, s_freq, window_size=512, overlap=0.50, n_ceps=13, post_process=False)     

# ax = sns.heatmap(mfcc1)
# ax.invert_yaxis()


    

