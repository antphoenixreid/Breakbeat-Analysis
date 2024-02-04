import sys
import numpy as np
from scipy import signal, ndimage
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import librosa

# Plot Matrix Colorbar
def plot_matrix(X, Fs=1, Fs_F=1, T_coef=None, F_coef=None, xlabel='Time (seconds)', ylabel='Frequency (Hz)',
                xlim=None, ylim=None, clim=None, title='', dpi=72,
                colorbar=True, colorbar_aspect=20.0, cbar_label='', ax=None, figsize=(6, 3), **kwargs):
    """2D raster visualization of a matrix, e.g. a spectrogram or a tempogram.

    Args:
        X: The matrix
        Fs: Sample rate for axis 1 (Default value = 1)
        Fs_F: Sample rate for axis 0 (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        F_coef: Frequency coeffients. If None, will be computed, based on Fs_F. (Default value = None)
        xlabel: Label for x-axis (Default value = 'Time (seconds)')
        ylabel: Label for y-axis (Default value = 'Frequency (Hz)')
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        clim: Color limits (Default value = None)
        title: Title for plot (Default value = '')
        dpi: Dots per inch (Default value = 72)
        colorbar: Create a colorbar. (Default value = True)
        colorbar_aspect: Aspect used for colorbar, in case only a single axes is used. (Default value = 20.0)
        cbar_label: Label for colorbar (Default value = '')
        ax: Either (1.) a list of two axes (first used for matrix, second for colorbar), or (2.) a list with a single
            axes (used for matrix), or (3.) None (an axes will be created). (Default value = None)
        figsize: Width, height in inches (Default value = (6, 3))
        **kwargs: Keyword arguments for matplotlib.pyplot.imshow

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = [ax]
    if T_coef is None:
        T_coef = np.arange(X.shape[1]) / Fs
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F

    if 'extent' not in kwargs:
        x_ext1 = (T_coef[1] - T_coef[0]) / 2
        x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
        y_ext1 = (F_coef[1] - F_coef[0]) / 2
        y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
        kwargs['extent'] = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'inferno'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    im = ax[0].imshow(X, **kwargs)

    if len(ax) == 2 and colorbar:
        cbar = plt.colorbar(im, cax=ax[1])
        cbar.set_label(cbar_label)
    elif len(ax) == 2 and not colorbar:
        ax[1].set_axis_off()
    elif len(ax) == 1 and colorbar:
        plt.sca(ax[0])
        cbar = plt.colorbar(im, aspect=colorbar_aspect)
        cbar.set_label(cbar_label)

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    if xlim is not None:
        ax[0].set_xlim(xlim)
    if ylim is not None:
        ax[0].set_ylim(ylim)
    if clim is not None:
        im.set_clim(clim)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, im

# Plot Chromagram
def plot_chromagram(*args, chroma_yticks=np.arange(12), **kwargs):
    """Call libfmp.b.plot_matrix and sets chroma labels.

    See :func:`libfmp.b.b_plot.plot_matrix` for parameters and return values.
    """
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Chroma'
    fig, ax, im = plot_matrix(*args, **kwargs)

    chroma_names = 'C C# D D# E F F# G G# A A# B'.split()
    ax[0].set_yticks(np.array(chroma_yticks))
    ax[0].set_yticklabels([chroma_names[i] for i in chroma_yticks])

    return fig, ax, im

# Smooth Downsampling
def smooth_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10, w_type='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel

    Args:
        X (np.ndarray): Feature sequence
        Fs (scalar): Frame rate of ``X``
        filt_len (int): Length of smoothing filter (Default value = 41)
        down_sampling (int): Downsampling factor (Default value = 10)
        w_type (str): Window type of smoothing filter (Default value = 'boxcar')

    Returns:
        X_smooth (np.ndarray): Smoothed and downsampled feature sequence
        Fs_feature (scalar): Frame rate of ``X_smooth``
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature

def compute_local_average(x, M):
    """Compute local average of signal

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100.0, M=10, norm=True):
    """Compute spectral-based novelty function

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 256)
        gamma (float): Parameter for logarithmic compression (Default value = 100.0)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_spectrum (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

    Notebook: C3/C3S1_FeatureNormalization.ipynb

    Args:
        X (np.ndarray): Feature sequence
        norm (str): The norm to be applied. '1', '2', 'max' or 'z' (Default value = '2')
        threshold (float): An threshold below which the vector ``v`` used instead of normalization
            (Default value = 0.0001)
        v (float): Used instead of normalization below ``threshold``. If None, uses unit vector for given norm
            (Default value = None)

    Returns:
        X_norm (np.ndarray): Normalized feature sequence
    """
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm

def resample_signal(x_in, Fs_in, Fs_out=100, norm=True, time_max_sec=None, sigma=None):
    """Resample and smooth signal

    Notebook: C6/C6S1_NoveltyComparison.ipynb

    Args:
        x_in (np.ndarray): Input signal
        Fs_in (scalar): Sampling rate of input signal
        Fs_out (scalar): Sampling rate of output signal (Default value = 100)
        norm (bool): Apply max norm (if norm==True) (Default value = True)
        time_max_sec (float): Duration of output signal (given in seconds) (Default value = None)
        sigma (float): Standard deviation for smoothing Gaussian kernel (Default value = None)

    Returns:
        x_out (np.ndarray): Output signal
        Fs_out (scalar): Feature rate of output signal
    """
    if sigma is not None:
        x_in = ndimage.gaussian_filter(x_in, sigma=sigma)
    T_coef_in = np.arange(x_in.shape[0]) / Fs_in
    time_in_max_sec = T_coef_in[-1]
    if time_max_sec is None:
        time_max_sec = time_in_max_sec
    N_out = int(np.ceil(time_max_sec*Fs_out))
    T_coef_out = np.arange(N_out) / Fs_out
    if T_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        T_coef_in = np.append(T_coef_in, [T_coef_out[-1]])
    x_out = interp1d(T_coef_in, x_in, kind='linear')(T_coef_out)
    if norm:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out, Fs_out

def compute_tempogram_fourier(x, Fs, N, H, Theta=np.arange(30, 601, 1)):
    """Compute Fourier-based tempogram [FMP, Section 6.2.2]

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601, 1))

    Returns:
        X (np.ndarray): Tempogram
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
    """
    win = np.hanning(N)
    N_left = N // 2
    L = x.shape[0]
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    # x_pad = np.pad(x, (L_left, L_right), 'constant')  # doesn't work with jit
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    t_pad = np.arange(L_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    K = len(Theta)
    X = np.zeros((K, M), dtype=np.complex_)

    for k in range(K):
        omega = (Theta[k] / 60) / Fs
        exponential = np.exp(-2 * np.pi * 1j * omega * t_pad)
        x_exp = x_pad * exponential
        for n in range(M):
            t_0 = n * H
            t_1 = t_0 + N
            X[k, n] = np.sum(win * x_exp[t_0:t_1])
        T_coef = np.arange(M) * H / Fs
        F_coef_BPM = Theta
    return X, T_coef, F_coef_BPM

def compute_cyclic_tempogram(tempogram, F_coef_BPM, tempo_ref=30,
                             octave_bin=40, octave_num=4):
    """Compute cyclic tempogram

    Args:
        tempogram (np.ndarray): Input tempogram
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
        tempo_ref (float): Reference tempo (BPM) (Default value = 30)
        octave_bin (int): Number of bins per tempo octave (Default value = 40)
        octave_num (int): Number of tempo octaves to be considered (Default value = 4)

    Returns:
        tempogram_cyclic (np.ndarray): Cyclic tempogram tempogram_cyclic
        F_coef_scale (np.ndarray): Tempo axis with regard to scaling parameter
        tempogram_log (np.ndarray): Tempogram with logarithmic tempo axis
        F_coef_BPM_log (np.ndarray): Logarithmic tempo axis (BPM)
    """
    F_coef_BPM_log = tempo_ref * np.power(2, np.arange(0, octave_num*octave_bin)/octave_bin)
    F_coef_scale = np.power(2, np.arange(0, octave_bin)/octave_bin)
    tempogram_log = interp1d(F_coef_BPM, tempogram, kind='linear', axis=0, fill_value='extrapolate')(F_coef_BPM_log)
    K = len(F_coef_BPM_log)
    tempogram_cyclic = np.zeros((octave_bin, tempogram.shape[1]))
    for m in np.arange(octave_bin):
        tempogram_cyclic[m, :] = np.mean(tempogram_log[m:K:octave_bin, :], axis=0)
    return tempogram_cyclic, F_coef_scale, tempogram_log, F_coef_BPM_log

def compute_autocorrelation_local(x, Fs, N, H, norm_sum=True):
    """Compute local autocorrelation

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = True)

    Returns:
        A (np.ndarray): Time-lag representation
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_lag (np.ndarray): Lag axis
    """
    # L = len(x)
    L_left = round(N / 2)
    L_right = L_left
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    L_pad = len(x_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    A = np.zeros((N, M))
    win = np.ones(N)
    if norm_sum is True:
        lag_summand_num = np.arange(N, 0, -1)
    for n in range(M):
        t_0 = n * H
        t_1 = t_0 + N
        x_local = win * x_pad[t_0:t_1]
        r_xx = np.correlate(x_local, x_local, mode='full')
        r_xx = r_xx[N-1:]
        if norm_sum is True:
            r_xx = r_xx / lag_summand_num
        A[:, n] = r_xx
    Fs_A = Fs / H
    T_coef = np.arange(A.shape[1]) / Fs_A
    F_coef_lag = np.arange(N) / Fs
    return A, T_coef, F_coef_lag

def compute_tempogram_autocorr(x, Fs, N, H, norm_sum=False, Theta=np.arange(30, 601)):
    """Compute autocorrelation-based tempogram

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = False)
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601))

    Returns:
        tempogram (np.ndarray): Tempogram tempogram
        T_coef (np.ndarray): Time axis T_coef (seconds)
        F_coef_BPM (np.ndarray): Tempo axis F_coef_BPM (BPM)
        A_cut (np.ndarray): Time-lag representation A_cut (cut according to Theta)
        F_coef_lag_cut (np.ndarray): Lag axis F_coef_lag_cut
    """
    tempo_min = Theta[0]
    tempo_max = Theta[-1]
    lag_min = int(np.ceil(Fs * 60 / tempo_max))
    lag_max = int(np.ceil(Fs * 60 / tempo_min))
    A, T_coef, F_coef_lag = compute_autocorrelation_local(x, Fs, N, H, norm_sum=norm_sum)
    A_cut = A[lag_min:lag_max+1, :]
    F_coef_lag_cut = F_coef_lag[lag_min:lag_max+1]
    F_coef_BPM_cut = 60 / F_coef_lag_cut
    F_coef_BPM = Theta
    tempogram = interp1d(F_coef_BPM_cut, A_cut, kind='linear',
                         axis=0, fill_value='extrapolate')(F_coef_BPM)
    return tempogram, T_coef, F_coef_BPM, A_cut, F_coef_lag_cut

# Main Code Block
if __name__ == "__main__":
    # Prepare audio file
    if len(sys.argv) > 1:
        song_file = "E:/Engineering/Signal Processing/Personal Projects/Breakbeat Analysis/Data/" + sys.argv[1]
        print(song_file)
        Fs = 22050
        x, Fs = librosa.load(song_file, Fs)
        x_dur = x.shape[0]/Fs
        
        # Chromagram
        N, H = 2048, 1024
        chromagram = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
        
        filt_len = 41
        down_sampling = 10
        filt_kernel = np.ones([1,filt_len])
        chromagram_smooth =  signal.convolve(chromagram, filt_kernel, mode='same')/filt_len
        chromagram_smooth = chromagram_smooth[:,::down_sampling]
        chromagram_smooth, Fs_smooth = smooth_downsample_feature_sequence(chromagram, Fs/H, filt_len=filt_len, down_sampling=down_sampling)
        
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                          'height_ratios': [2]}, figsize=(14, 5))
        plot_chromagram(chromagram_smooth, Fs=Fs_smooth, ax=[axs[0], axs[1]],  
                          chroma_yticks = [0,4,7,11], 
                          title='Chromagram (resolution %0.1f Hz)'%(Fs_smooth), 
                          ylabel='Chroma', colorbar=True)
        
        # MFCC
        X_MFCC = librosa.feature.mfcc(y=x, sr=Fs, hop_length=H, n_fft=N)
        coef = np.arange(4, 15)
        X_MFCC_upper = X_MFCC[coef, :]
        
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                          'height_ratios': [2]}, figsize=(14, 5))
        plot_matrix(X_MFCC_upper, Fs=Fs/H, ax=[axs[0], axs[1]],
                    title='MFCC (coeffiecients 4 to 14)', ylabel='', colorbar=True)
        
        # Fourier Tempogram
        nov, Fs_nov = compute_novelty_spectrum(x, Fs=Fs, N=N, H=H, gamma=100, M=10, norm=True)
        nov, Fs_nov = resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)
        
        X, T_coef, F_coef_BPM = compute_tempogram_fourier(nov, Fs_nov, N=1000, H=100, Theta=np.arange(30, 601))
        
        octave_bin = 30
        tempogram_F = np.abs(X)
        output = compute_cyclic_tempogram(tempogram_F, F_coef_BPM, octave_bin=octave_bin)
        X_FT = output[0]
        F_coef_scale = output[1]
        X_FT = normalize_feature_sequence(X_FT, norm='max')
        
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                          'height_ratios': [2]}, figsize=(14, 5))
        plot_matrix(X_FT, T_coef=T_coef, ax=[axs[0], axs[1]], 
                      title='Fourier-based cyclic tempogram', ylabel='Scaling',
                      colorbar=True, clim=[0, 1])
        
        # Autocorrelation Tempogram
        output = compute_tempogram_autocorr(nov, Fs_nov, N=500, H=100, norm_sum=False,
                                              Theta=np.arange(30, 601))
        
        tempogram_A = output[0]
        T_coef = output[1]
        F_coef_BPM = output[2]
        
        output = compute_cyclic_tempogram(tempogram_A, F_coef_BPM, octave_bin=octave_bin)
        tempogram_cyclic_A = output[0]
        F_coef_scale = output[1]
        X_AT = normalize_feature_sequence(tempogram_cyclic_A, norm='max')
        
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                          'height_ratios': [2]}, figsize=(14, 5))
        plot_matrix(X_AT, T_coef=T_coef, ax=[axs[0], axs[1]], 
                     title='Autocorrelation-based cyclic tempogram', ylabel='Scaling',
                     colorbar=True, clim=[0, 1])
        
        plt.show()
    else:
        print("Missing audio file to analyze...")