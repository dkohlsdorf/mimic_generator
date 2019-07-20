import os

from sys import argv
from mimic_utils.mimic_generators import *
from scipy.io import wavfile

fs, data_click = wavfile.read('resources/one_click.wav')

def header():
        return """
        Mimimc Generation Script

        Usage: 
                python mkmimics.py INFOLDER OUTFOLDER NWHISTLES

        What Will Happen:
                For each of the wav files in the INFOLDER, 
                the script will generate burst based whistles, 
                simple mimics by shifting and also anti whistles.
                In each of these categories we generate NWHISTLES
                randomly and add them into the outfolder

        by Daniel Kohlsdorf
        """        

def params():
        if len(argv) != 4:
                return None
        else:
                infolder  = argv[1]
                outfolder = argv[2]
                n_mimics  = int(argv[3])
                return infolder, outfolder, n_mimics

if __name__ == "__main__":
        print(header())        
        p = params()        
        if p != None: 
                infolder, outfolder, n_mimics = p
                print("Reading From {}".format(infolder))
                print("Writing To   {}".format(outfolder))
                print("Generating:  {}".format(n_mimics))
                
                '''
                for filename in os.listdir(infolder):
                    if filename.endswith(".wav"):
                        print("Processing: {} to burst".format(filename))
                        basename = filename.replace(".wav", "")
                        infile   = "{}/{}.wav".format(infolder, basename)
                        outfile  = "{}/{}_burst.wav".format(outfolder, basename)        
                        fs, data_whistle = wavfile.read(infile)    
                        audio            = mimic_burst(data_click, data_whistle)
                        print("\t3) Write Result")
                        wavfile.write(outfile, fs, audio * 0.001)
                '''

                for filename in os.listdir(infolder):
                    if filename.endswith(".wav"):
                        for i in range(0, n_mimics):
                                print("Processing: {} to mimic".format(filename))
                                basename = filename.replace(".wav", "")
                                infile   = "{}/{}.wav".format(infolder, basename)
                                outfile  = "{}/{}_mimic_{}.wav".format(outfolder, basename, i)        
                                fs, data_whistle = wavfile.read(infile)    
                                audio            = mimic(data_whistle, fs, len(data_whistle) / fs)
                                print("\t3) Write Result")
                                wavfile.write(outfile, fs, audio * 0.1)

                for filename in os.listdir(infolder):
                    if filename.endswith(".wav"):
                        for i in range(0, n_mimics):
                                print("Processing: {} to click".format(filename))
                                basename = filename.replace(".wav", "")
                                infile   = "{}/{}.wav".format(infolder, basename)
                                outfile  = "{}/{}_click_{}.wav".format(outfolder, basename, i)        
                                fs, data_whistle = wavfile.read(infile)    
                                deleted          = mimic_by_clicks(data_click, data_whistle)
                                print("\t3) Write Result")
                                wavfile.write(outfile, fs, deleted * 0.001)
                                print("\tDone")
