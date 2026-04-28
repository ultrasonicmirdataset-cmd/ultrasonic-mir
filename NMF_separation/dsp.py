"""
dsp.py
Purpose: Digital signal processing utilities including STFT computation, spectrogram plotting, and audio reconstruction.
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np

def compute_stft(y, n_fft=1024, hop_length=512, win_length=None, window="hann"):
    if win_length is None:
        win_length = n_fft
    
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

def compute_istft(D, hop_length=512, win_length=None, length=None, window="hann"):
    if win_length is None:
        win_length = 2 * (D.shape[0] - 1)
        
    return librosa.istft(D, hop_length=hop_length, win_length=win_length, length=length, window=window)

def audio_to_V(filepath, n_fft=1024, hop_length=512, window="hann", start_sec=None, end_sec=None):
    y, sr = librosa.load(filepath, sr=None, offset=start_sec, duration=(end_sec - start_sec) if end_sec else None)
    D = compute_stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    V = np.abs(D)
    phase = np.exp(1.j * np.angle(D))
    return V, phase, sr, len(y)

def V_to_audio(W, H, phase, hop_length=512, length=None, window="hann"):
    V_reconstructed = W @ H
    D_reconstructed = V_reconstructed * phase
    return compute_istft(D_reconstructed, hop_length=hop_length, length=length, window=window)

def plot_spectrogram(D, sr, hop_length, title="Spectrogram", y_axis='linear'):
    plt.figure(figsize=(8, 4))
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.colorbar(label="dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()
