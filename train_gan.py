import matplotlib.pyplot as plt

from sys import argv
from mimic_utils.nn import *
from mimic_utils.data_utils import *
from mimic_utils.params import *
from scipy.io import wavfile

def header():
    return """
    Mimic Geberation Scripts using Generative Adversarial Neural Netorks 

    Usage:
         python train_gan.py WHISTLE_FOLDER TARGET_FOLDER OUTFOLDER LEN EPOCH GEN

    What Will Happen:
         For each file in the WHISTLE_FOLDER we search associated outputs
         in the TARGET_FOLDER and train a GAN from whistle traces + noise 
         to the targets. The results are written in the OUTFOLDER. We padd / cut
         all traces and spectrogram to the desired LEN. We train the net for
         EPOCHs and then generate GEN examples for each trace.
         
    by Daniel Kohlsdorf
    """

def params():
    if len(argv) != 7:
        return None
    else:
        infolder      = argv[1]
        targetfolder  = argv[2]
        outfolder     = argv[3]
        length        = int(argv[4])
        epochs        = int(argv[5])
        n_mimics      = int(argv[6])
        return infolder, targetfolder, outfolder, length, epochs, n_mimics


if __name__ == '__main__':
    print(header())
    p = params()
    if p != None:
        infolder, targetfolder, outfolder, n, epochs, n_mimics = p
        whistle_traces  = traces(infolder, -1, n)
        whistle_targets = target_spectrograms(targetfolder, whistle_traces.keys(), n)
        data_trace, data_spec = create_dataset(whistle_traces, whistle_targets)

        gan = GAN(length = n)
        gan.train(data_trace, data_spec, epochs)
        
        for basename, trace in whistle_traces.items():
            for i in range(n_mimics):
                audio       = gan.create(trace, FFT_WIN, FFT_STEP)
                outfile     = "{}/{}_gan_{}.wav".format(outfolder, basename, i)
                wavfile.write(outfile, 190000, audio * 10)
