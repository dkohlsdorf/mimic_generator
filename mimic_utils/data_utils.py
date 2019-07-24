import os

from mimic_utils.spectrogram import *
from mimic_utils.whistle_tracer import *
from mimic_utils.params import *

from scipy.io import wavfile


def target_spectrograms(folder, keys):
    pass


def traces(folder, max_whistles):
    '''
    Read all whistles from a folder

    folder: a folder containing traceable whistles as wav files
    max_whistles: only read that many 
    '''
    traces = {}
    for filename in os.listdir(folder):
        if max_whistles < 0 or len(traces) <= max_whistles:
            if filename.endswith('.wav'):
                print("Traceing: {}".format(filename))
                basename = filename.replace(".wav", "")
                infile   = "{}/{}.wav".format(folder, basename)
                fs, data_whistle = wavfile.read(infile)            
                spec_whistle = fwd_spectrogram(data_whistle[:, 0], FFT_WIN, FFT_STEP) 
                whistle_trace, _  = trace(spec_whistle, TRACE_RAD, SMOOTH_ENT)    
                traces[basename] = normalize(whistle_trace[:-SMOOTH_ENT])
                print("Trace Length: {} ".format(whistle_trace.shape))
    return traces


def stack_whistles(whistles, max_len):
    '''
    Build a matrix of all whistle traces
    
    whistles: a list of traces
    max_len: maximum length
    '''
    n = len(whistles)
    stack = np.zeros((n, max_len))
    for i, whistle in enumerate(whistles):
        t = min(len(whistle), max_len)
        stack[i, 0:t] = whistle[0:t]
    return stack



def normalize(trace):
    '''
    Normalize a whistle trace
    
    trace: the whistle trace
    '''
    lo = min(trace)
    hi = max(trace)
    return (trace - lo) / (hi - lo)

