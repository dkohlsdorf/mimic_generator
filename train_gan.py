import matplotlib.pyplot as plt

from sys import argv
from mimic_utils.nn import *
from mimic_utils.data_utils import *
from mimic_utils.params import *
from scipy.io import wavfile

whistle_traces  = traces("whistles", 1, 1000)
whistle_targets = target_spectrograms("target_output", whistle_traces.keys(), 1000)
data_trace, data_spec = create_dataset(whistle_traces, whistle_targets)

gan = GAN(length = 1000)
gan.train(data_trace, data_spec, 1)
gan.save(".")

for basename, trace in whistle_traces.items():
    for i in range(10):
        audio       = gan.create(trace, FFT_WIN, FFT_STEP)
        outfile     = "output/{}_gan_{}.wav".format(basename, i)
        outfile_txt = "output/{}_gan_{}.txt".format(basename, i)
        wavfile.write(outfile, 190000, audio * 10)
        np.savetxt(outfile_txt, audio)
