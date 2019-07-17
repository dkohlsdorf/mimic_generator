from mimic_utils.mimic_generators import *
from scipy.io import wavfile

import os

fs, data_click = wavfile.read('resources/one_click.wav')
for filename in os.listdir("whistles/"):
    if filename.endswith(".wav"):
        print("Processing: {}".format(filename))
        basename = filename.replace(".wav", "")
        infile   = "whistles/{}.wav".format(basename)
        outfile  = "output/{}.wav".format(basename)        
        fs, data_whistle = wavfile.read(infile)    
        deleted = mimic_by_clicks(data_click, data_whistle)
        print("\t3) Write Result")
        wavfile.write(outfile, fs, deleted * 0.001)
        print("\tDone")
