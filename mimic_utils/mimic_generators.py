from mimic_utils.spectrogram import *
from mimic_utils.whistle_tracer import * 
from mimic_utils.mimic import * 

import matplotlib.pyplot as plt

# Dynamics Parameters
HARMONICS  = 5
MARGIN     = 8
SHIFT      = 5

# Fixed Params
FFT_WIN    = 1024
FFT_STEP   = 512
TRACE_RAD  = 10
SMOOTH_ENT = 64
LOUDNESS   = 1.0


def mimic_by_clicks(one_click, data_whistle):
    '''
    Generate a mimic of a whistle using clicks and the anti whistle

    one_click: one channel raw audio of exactly one click
    data_whistle: raw audio of the whistle
    outfile: filename for the result
    fs: sample rate of the result
    ''' 
    print("\t1) Compute Spectrograms")
    spec_whistle = fwd_spectrogram(data_whistle[:, 0], FFT_WIN, FFT_STEP)    

    print("\t2) Trace Whistle And Create Mask")
    whistle_trace, _           = trace(spec_whistle, TRACE_RAD, SMOOTH_ENT)
    data_click, delay_profile  = mk_click(whistle_trace, one_click[:, 0], data_whistle[:, 0].shape[0])
    spec_click                 = fwd_spectrogram(data_click, FFT_WIN, FFT_STEP)
    whistle_mask               = delete_mask(whistle_trace, spec_click.shape[0], FFT_WIN, MARGIN, HARMONICS, SHIFT)
    deleted                    = bwd_spectrogram(spec_click * whistle_mask, FFT_WIN, FFT_STEP)    
    return deleted
