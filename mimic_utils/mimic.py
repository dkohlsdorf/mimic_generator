import numpy as np
import random

def slices(trace):
    '''
    Cut a trace into active regions (non zero)

    trace: A whsitle trace with inactive regions set to zero
    '''
    slices =[]
    current_slice = []
    for freq in trace:
        if freq == 0.0 and len(current_slice) > 0:
            slices.append(current_slice)
            current_slice = []
        else:
            current_slice.append(freq)
    return slices    


def build_whistle(slices, target_len):
    '''
    Build a whistle from whistle slices by sampling gap lengths

    target_length: length of the output
    '''
    slice_len = np.sum([len(slice) for slice in slices])
    gap_len   = target_len - slice_len
    n_gaps    = len(slices) + 1
    gaps      = random.shuffle(np.arange(n_gaps))
    steps     = ((1.0 / np.dot(gaps, gaps)) * gaps) * target_len # think linear regression
    fillers   = [np.zeros(step[i] * gaps[i]) for i in range(0, n_gaps)]
    output    = np.zeros(target_len)

    end = len(fillers[0])
    output[0:end] = fillers[0]
    for i in range(0, len(slices)):
        start = end
        end   = end + len(slices[i])
        output[start:end] = slices[i]
        start = end
        end   = end + len(fillers[i + 1])
        output[start:end] = fillers[i]
    return output


def scale_up(trace, by):
    '''
    Warp a whistle to be longer

    trace: a whistle's trace
    by: scaling factor
    '''
    assert by > 1
    return np.repeat(whistle, by)


def scale_down(trace, by):
    '''
    Warp a whistle to be shorter

    trace: a whistle's trace
    by: scaling factor
    '''
    assert by > 1
    n = len(trace)
    return trace[np.arange(n) % by == 0]     


def trace2spectrogram(path, loudness, w, h, margin, harmonics, shift_up):
    '''
    Build a spectrogram from a whistle's trace

    path: a whistle trace (sequence of frequency bins)
    loudness: the loudness of each element in the trace
    w: width of mask
    h: height of mask
    margin: how thick to paint the whistle
    harmonics: number of harmonics from trace
    shift_up: number of bins to shift the whistle
    '''
    n = len(path)    
    mask = np.ones((np.max(w, n) , h))
    for t in range(0, n):
        if path[t] > 0.0:  
            for harmonic in range(1, harmonics + 1):
                start = int(max(path[t] * harmonic - margin + shift_up, 0)) 
                stop  = int(min(path[t] * harmonic + margin + shift_up, h / 2))
                for f in range(start, stop):
                    mask[t][f]     = loudness[t]
                    mask[t][h - f] = loudness[t]
    return mask



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
    