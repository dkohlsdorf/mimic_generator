from mimic_utils.spectrogram import *
from mimic_utils.whistle_tracer import * 
from mimic_utils.mimic import * 
from mimic_utils.params import *


def mimic_by_clicks(one_click, data_whistle):
    '''
    Generate a mimic of a whistle using clicks and the anti whistle

    one_click: one channel raw audio of exactly one click
    data_whistle: raw audio of the whistle
    ''' 
    print("\t1) Compute Spectrograms")
    spec_whistle = fwd_spectrogram(data_whistle[:, 0], FFT_WIN, FFT_STEP)    
    print("\t2) Trace Whistle And Create Mask")
    whistle_trace, _  = trace(spec_whistle, TRACE_RAD, SMOOTH_ENT)    
    data_click, _     = mk_click(whistle_trace, one_click[:, 0], data_whistle[:, 0].shape[0], FFT_WIN)
    spec_click        = fwd_spectrogram(data_click, FFT_WIN, FFT_STEP)
    whistle_mask      = delete_mask(whistle_trace, spec_click.shape[0], FFT_WIN, MARGIN.sample(), HARMONICS.sample(), SHIFT.sample())
    deleted           = bwd_spectrogram(spec_click * whistle_mask, FFT_WIN, FFT_STEP)    
    return deleted


def mimic(data_whistle, rate, seconds):
    '''
    Generates a mimic shifted in frequency.

    data_whistle: the whistle to generate from
    rate: the sample rate of the output
    seconds: desired number of seconds
    '''
    print("\t1) Compute Spectrograms")
    spec_whistle = fwd_spectrogram(data_whistle[:, 0], FFT_WIN, FFT_STEP)    
    print("\t2) Trace Whistle And Create Audio")
    whistle_trace, _ = trace(spec_whistle, TRACE_RAD, SMOOTH_ENT)
    whistle_trace   *= SHIFT.sample()
    frequencies      = trace2frequencies(whistle_trace[:-50], rate, FFT_WIN)
    audio            = trace2audio(frequencies, rate, seconds)    
    return np.array(audio)


def mimic_burst(one_click, data_whistle):
    '''
    Create a burst from a whistle

    one_click: one channel raw audio of exactly one click
    data_whistle: raw audio of the whistle
    '''
    print("\t1) Compute Spectrograms")
    spec_whistle = fwd_spectrogram(data_whistle[:, 0], FFT_WIN, FFT_STEP)    
    print("\t2) Trace Whistle And Create Audio")
    whistle_trace, _ = trace(spec_whistle, TRACE_RAD, SMOOTH_ENT)
    audio            = trace2burst(whistle_trace, one_click[:, 0], data_whistle[:, 0].shape[0])
    return audio