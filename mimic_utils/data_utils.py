import os

from mimic_utils.spectrogram import *
from mimic_utils.whistle_tracer import *
from mimic_utils.params import *

from scipy.io import wavfile


def create_dataset(traces, targets):
    '''
    Create a dataset of associated inputs and outputs
    
    traces: a dictionary of traces, basename from file is the key
    targets: a dictionary of spectrogram lists  
    '''
    normalized_traces = []
    normalized_specs  = [] 
    for (basename, trace) in traces.items():
        for spectrogram in targets[basename]:
            (t, d) = spectrogram.shape
            trace = trace.reshape(1, t)
            spectrogram = spectrogram.reshape(1, t, d, 1)
            normalized_traces.append(trace)
            normalized_specs.append(spectrogram)
    return normalized_traces, normalized_specs


def target_spectrograms(folder, keys, max_len):
    '''
    Read spectrograms for target output 

    folder: Folder containing the audio for the target spectrograms
    keys: the base names allowed
    max_len: maximum length 
    '''
    specs = {}
    for key in keys:
        specs[key] = []
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):            
            basename = None
            for key in keys:
                if key in filename:
                    basename = key            
            infile = "{}/{}".format(folder, filename)
            fs, data = wavfile.read(infile)            
            spec_whistle = fwd_spectrogram(data, FFT_WIN, FFT_STEP)             
            t = min(len(spec_whistle), max_len)
            dim = int(FFT_WIN / 2)
            padded = np.zeros((max_len, dim))
            padded[0:t, :] = np.abs(spec_whistle[0:t, 0:dim])
            specs[key].append(padded)
    return specs


def traces(folder, max_whistles, max_len):
    '''
    Read all whistles from a folder

    folder: a folder containing traceable whistles as wav files
    max_whistles: only read that many 
    max_len: maximum length of the whistle, if the whistle is shorted it is padded     
    '''
    traces = {}
    for filename in os.listdir(folder):
        if max_whistles < 0 or len(traces) < max_whistles:
            if filename.endswith('.wav'):
                print("Traceing: {}".format(filename))
                basename = filename.replace(".wav", "")
                infile   = "{}/{}.wav".format(folder, basename)
                fs, data_whistle = wavfile.read(infile)            
                spec_whistle = fwd_spectrogram(data_whistle[:, 0], FFT_WIN, FFT_STEP) 
                whistle_trace, _  = trace(spec_whistle, TRACE_RAD, SMOOTH_ENT)
                t = min(len(whistle_trace) - SMOOTH_ENT, max_len)
                padded = np.zeros(max_len)
                padded[0:t] = normalize(whistle_trace[:-SMOOTH_ENT])[0:t]
                traces[basename] = padded
                print("Trace Length: {} ".format(whistle_trace.shape))
    return traces

def normalize(trace):
    '''
    Normalize a whistle trace
    
    trace: the whistle trace
    '''
    lo = min(trace)
    hi = max(trace)
    return (trace - lo) / (hi - lo)


