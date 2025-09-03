from matplotlib import pyplot as plt
import numpy as np
from numba import jit
import librosa

from audio_process import Audio_Process

# @jit(nopython=True)
# def f_pitch(p, pitch_ref=69, freq_ref=440.0):
#     return 2**((p - pitch_ref)/12)*freq_ref

# @jit(nopython=True)
# def pool_pitch(Fs, N, p, pitch_ref=69, freq_ref=440.0):
#     lower = f_pitch(p - 0.5, pitch_ref=pitch_ref, freq_ref=freq_ref)
#     upper = f_pitch(p + 0.5, pitch_ref=pitch_ref, freq_ref=freq_ref)

#     k = np.arange(N//2 + 1)
#     k_freq = k*Fs/N
#     mask = np.logical_and(lower <= k_freq, k_freq < upper)

#     return k[mask]

# @jit(nopython=True)
# def compute_spec_log_freq(X, Fs, N):
#     Y_LF = np.zeros((128, X.shape[1]), dtype=np.float32)

#     for p in range(128):
#         k = pool_pitch(p, N)
#         Y_LF[p, :] = X[k, :].sum(axis=0)

#     F_coef_pitch = np.arange(128)

#     return Y_LF, F_coef_pitch

def main():
    audio_file = "D:/Engineering/Signal Processing/Personal Projects/Breakbeat Analysis/Data/Funky Drummer-James Brown/James Brown - Funky Drummer (Full Version 1970) - HQ.mp3"
    x, Fs = librosa.load(audio_file, sr=22050)
    N = 2048
    H = 512

    processor = Audio_Process(x, Fs, N=N, H=H, mag=True)
    eps = np.finfo(float).eps

    # fig = plt.figure(figsize=(16, 12))
    # ax = plt.subplot(1, 1, 1)

    # time = np.arange(x.shape[0])/Fs

    # ax.plot(time, x)
    # ax.set_xlim([time[0], time[-1]])
    # y_lim_x = x[np.isfinite(x)]
    # x_min, x_max = y_lim_x.min(), y_lim_x.max()
    # if x_max == x_min:
    #     x_max = x_max + 1
    # ax.set_ylim([min(1.1*x_min, 0.9*x_min), max(1.1*x_max, 0.9*x_max)])
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Applitude')
    # plt.tight_layout()
    # plt.show()

    # fig = plt.figure(figsize=(16, 12))

    x_ext1 = (processor.T_coef[1] - processor.T_coef[0])/2
    x_ext2 = (processor.T_coef[-1] - processor.T_coef[-2])/2
    y_ext1 = (processor.F_coef[1] - processor.F_coef[0])/2
    y_ext2 = (processor.F_coef[-1] - processor.F_coef[-2])/2
    extent = [processor.T_coef[0] - x_ext1, processor.T_coef[-1] + x_ext2, 
              processor.F_coef[0] - y_ext1, processor.F_coef[-1] + y_ext2]
    
    # plt.imshow(processor.X, origin='lower', aspect='auto', cmap='hot', 
    #         extent=extent)
    # plt.clim([-30, 30])
    # plt.ylim([0, 4500])
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Frequency (Hz)')
    # cbar = plt.colorbar()
    # cbar.set_label('Magnitude (dB)')
    # plt.tight_layout()
    # plt.show()

    # Compute Log-Frequency Spectrogram
    X_LF, F_coef_pitch = processor.compute_spec_log_freq(N)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(10*np.log10(X_LF + eps), origin='lower', aspect='auto', cmap='hot',
               extent=[processor.T_coef[0] - x_ext1, processor.T_coef[-1] + x_ext2, 
                       F_coef_pitch[0] - 0.5, F_coef_pitch[-1] + 0.5])
    plt.clim([-10, 50])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (pitch)')
    cbar = plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Compute Chromagram
    chroma = processor.compute_chromagram(N)

    fig = plt.figure(figsize=(8, 6))
    chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    plt.imshow(10*np.log10(chroma + eps), origin='lower', aspect='auto', cmap='hot',
               extent=[processor.T_coef[0], processor.T_coef[-1], 0, 12])
    plt.clim([0, 60])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Chroma')
    cbar = plt.colorbar()
    cbar.set_label('Magnitude (dB)')
    plt.yticks(np.arange(12) + 0.5, chroma_label)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()