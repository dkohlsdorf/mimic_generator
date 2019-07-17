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

    
def mk_click(trace, click, target_len):    
    '''
    Construct a clicktrain with the frequency of clicks changeing in
    proportion to a whistle trace.

    trace: a whistle's trace
    click: a click from a dolphin as one channel raw audio
    target_len: descired length of the clicktrain
    '''
    n = len(click)
    trace = np.array([512 - i for i in trace])
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
    