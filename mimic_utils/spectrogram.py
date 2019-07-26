import numpy as np
from numpy.fft import fft, ifft


def fwd_spectrogram(audio, win, step):
    '''
    Compute the spectrogram of audio data

    audio: one channel audio
    win: window size for dft sliding window
    step: step size for dft sliding windo
    '''
    spectrogram = []
    hanning = np.hanning(win)
    for i in range(win, len(audio), step):
        dft = fft(audio[i - win: i])
        spectrogram.append(dft)
    return np.array(spectrogram)
        
    
def bwd_spectrogram(spectrogram, win, step):
    '''
    Compute audio data from a spectrogram

    spectrogram: a spectrogram of audio data
    win: window size for dft sliding window
    step: step size for dft sliding windo
    '''
    t = len(spectrogram) * step + win
    audio  = np.zeros(t)
    scaler = np.ones(t)
    for (i, sample) in enumerate(spectrogram):
        window = ifft(sample)
        audio[i * step: (i * step) + win] += window.real 
        scaler[i * step: (i * step) + win] += 1
    return audio / scaler
