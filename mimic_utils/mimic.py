import numpy as np
import random


def delete_mask(path, w, h, margin, harmonics, shift_up):
    '''
    Build a mask to delete a whistle trace from a spectrogram

    path: a whistle trace (sequence of frequency bins)
    w: width of mask
    h: height of mask
    margin: how thick to paint the whistle
    harmonics: number of harmonics from trace
    shift_up: number of bins to shift the whistle
    '''
    mask = np.ones((w, h))
    n = len(path)    
    for t in range(0, n):
        if path[t] > 0.0:  
            for harmonic in range(1, harmonics + 1):
                start = int(max(path[t] * harmonic - margin + shift_up, 0)) 
                stop  = int(min(path[t] * harmonic + margin + shift_up, h / 2))
                for f in range(start, stop):
                    mask[t][f]     = 0.0
                    mask[t][h - f] = 0.0
    return mask

    
def mk_click(trace, click, target_len, fft_win):    
    '''
    Construct a clicktrain with the frequency of clicks changeing in
    proportion to a whistle trace.

    trace: a whistle's trace
    click: a click from a dolphin as one channel raw audio
    target_len: descired length of the clicktrain
    '''
    n = len(click)
    trace = np.array([fft_win / 2 - i for i in trace])
    steps = ((1.0 / np.dot(trace, trace)) * trace) * target_len # think linear regression
    click_train = np.zeros(target_len)
    scaler = np.ones(target_len)
    next_t = 0
    for i in range(0, len(trace)):      
        if next_t + n < target_len:
            click_train[next_t:next_t + n] += click
            scaler[next_t:next_t+n] += 1
        next_t += int(steps[i] * trace[i])            
    return click_train / scaler, steps


def trace2frequencies(trace, rate, fft_win):
    '''
    Compute the frequencies that relate to each frequency bin of the trace

    trace: a sequence of frequency bins
    rate: sample rate
    fft_win: window size
    '''
    dft_size = fft_win
    scaler   = rate / dft_size
    return trace * scaler


def interpolate(y1, y2, steps):
    '''
    Interpolte linearly between y1 and y2.

    y1: first value should be smallest
    y2: second value should be largest
    steps: number of steps to interpolate
    '''
    delta  = y2 - y1
    step   = delta / (steps - 1)
    linear = []
    for i in range(0, int(steps)):
        sample = y1 + (i * step)
        linear.append(sample)
    return linear


def trace2audio(trace, rate, sec):
    '''
    Write a trace as audio samples

    trace: a whistle trace as a sequence of frequencies
    rate: the sample rate
    sec: length of the audio in seconds  
    '''
    samples = rate * sec
    scale   = samples / len(trace)
    audio   = [0.0]
    t       = 0
    fade_in   = interpolate(0.05, 1.0, int(scale / 2))
    fade_out  = interpolate(1.0, 0.05, int(scale / 2))
    loudness  = fade_in + fade_out
    for i in range(len(trace)):     
        for j in range(0, len(loudness)):     
            raw  = np.sin(2.0 * np.pi * trace[i] * t / rate)    
            t += 1                            
            audio.append(raw * loudness[j])
    return np.array(audio)


def burst_packages(trace):
    '''
    Sets all regions in a trace that are larger than 1 to one.

    trace: a whistle trace
    '''
    trace[trace > 0.0] = 1.0
    return trace