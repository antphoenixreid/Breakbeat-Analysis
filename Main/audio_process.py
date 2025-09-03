import yt_dlp
import os, sys
import numpy as np
from numba import jit
import librosa
from matplotlib import pyplot as plt
from scipy.signal import stft, istft, find_peaks, get_window
from scipy.ndimage import median_filter
from scipy import ndimage, fftpack

class Audio_Process:
    # Initial Processing
    def __init__(self, audio_data, sample_rate, N=2048, H=512, pad_mode='constant', center=True, mag=False, gamma=0):

        # Process the data 
        self.y = audio_data
        self.sr = sample_rate

        self.X = librosa.stft(self.y, n_fft=N, win_length=N,
                              window='hann', center=center, pad_mode=pad_mode)
        
        if mag:
            self.X  = np.abs(self.X)**2

            if gamma > 0:
                self.X = np.log(1 + gamma*self.X)

        self.F_coef = np.arange(N//2 + 1)*self.sr/N
        self.T_coef = np.arange(self.X.shape[1])*H/self.sr

    # @jit(nopython=True)
    def f_pitch(self, p, pitch_ref=69, freq_ref=440.0):
        return 2**((p - pitch_ref)/12)*freq_ref
    
    # @jit(nopython=True)
    def pool_pitch(self, N, p, pitch_ref=69, freq_ref=440.0):
        lower = self.f_pitch(p - 0.5, pitch_ref=pitch_ref, freq_ref=freq_ref)
        upper = self.f_pitch(p + 0.5, pitch_ref=pitch_ref, freq_ref=freq_ref)

        k = np.arange(N//2 + 1)
        k_freq = k*self.sr/N
        mask = np.logical_and(lower <= k_freq, k_freq < upper)

        return k[mask]
    
    # @jit(nopython=True)
    def compute_spec_log_freq(self, N):
        Y_LF = np.zeros((128, self.X.shape[1]), dtype=np.float32)

        for p in range(128):
            k = self.pool_pitch(N, p)
            Y_LF[p, :] = self.X[k, :].sum(axis=0)

        F_coef_pitch = np.arange(128)

        return Y_LF, F_coef_pitch
    
    # @jit(nopython=True)
    def compute_chromagram(self, N):
        Y_LF, _ = self.compute_spec_log_freq(N)

        chroma = np.zeros((12, Y_LF.shape[1]), dtype=np.float32)
        p = np.arange(128)

        for c in range(12):
            mask = (p%12 == c)
            chroma[c, :] = Y_LF[mask, :].sum(axis=0)

        return chroma
    
    def note_name(p):
        chroma = ['A', 'A$^\\sharp$', 'B', 'C', 'C$^\\sharp$', 'D', 'D$^\\sharp$', 'E', 'F', 'F$^\\sharp$', 'G',
              'G$^\\sharp$']
        name = chroma[(p - 69) % 12] + str(p // 12 - 1)
        return name
    
    # @jit(nopython=True)
    def compute_mfccs(self, N, H, coef=np.arange(2, 14)):
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_fft=N, hop_length=H)

        if coef is not None:
            mfccs = mfccs[coef, :]

        return mfccs
    
    # @jit(nopython=True)
    def compute_tempogram(self, N, H, win_func='hann', theta=np.arange(30, 601, 1)):
        win = get_window(win_func, N)
        N_left = N//2
        L_left = L_right = N_left
        L_pad = L_left + L_right

        x_pad = np.concatenate((np.zeros(L_left), self.y, np.zeros(L_right)))
        t_pad = np.arange(L_pad)
        M = int(np.floor(L_pad - N)/H) + 1
        K = len(theta)
        X_BPM = np.zeros((K, M), dtype=np.complex_)

        for k in range(K):
            omega = (theta[k]/60)/self.sr
            exponential = np.exp(-2*np.pi*1j*omega*t_pad)
            x_exp = x_pad*exponential

            for m in range(M):
                t_0 = m*H
                t_1 = t_0 + N
                X_BPM[k, m] = np.sum(win*x_exp[t_0:t_1])

            T_coef_BPM = np.arange(M)*H/self.sr
            F_coef_BPM = theta

        return X_BPM, T_coef_BPM, F_coef_BPM
    
    def compute_local_average(self, M):
        L = len(self.x)
        local_average = np.zeros(L)

        for m in range(L):
            a = max(m - M, 0)
            b = min(m + M + 1, L)
            local_average[m] = (1/(2*M + 1))*np.sum(self.x[a:b])

        return local_average
    
    def compute_novelty_spectrum(self, N, H, gamma=100.0, M=10, norm=True):
        X = librosa.stft(self.y, n_fft=N, hop_length=H, win_length=N, window='hanning')
        Fs_feature = self.sr/H
        Y = np.log(1 + gamma*np.abs(X))
        Y_diff = np.diff(Y)
        Y_diff[Y_diff < 0] = 0
        novelty_spectrum = np.sum(Y_diff, axis=0)
        novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))

        if M > 0:
            local_average = self.compute_local_average(novelty_spectrum, M)
            novelty_spectrum = novelty_spectrum - local_average
            novelty_spectrum[novelty_spectrum < 0] = 0.0
        if norm:
            max_value = max(novelty_spectrum)
            if max_value > 0:
                novelty_spectrum = novelty_spectrum/max_value

        return novelty_spectrum, Fs_feature