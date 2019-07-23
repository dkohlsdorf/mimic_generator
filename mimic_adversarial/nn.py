import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *


def wasserstein_loss(y_true, y_pred):
    '''
    Wasserstein loss    
    '''
    return -K.mean(y_true * y_pred)


def generator(trace_len):
    '''
    Generate a spectrogram from a trace.
    The spectrogram will have 512 frequency bins and
    be exactly as long as the trace.

    trace_len: Length of the trace
    '''
    input_dim  = trace_len * 2
    input = Input(shape=(input_dim), name = "Trace")    
    x     = Reshape((trace_len, 1, 2))(input)
    x     = UpSampling2D(size = (1, 4))(x)
    x     = Conv2D(32,  kernel_size = (3, 3), padding='same', activation='relu')(x)
    x     = UpSampling2D(size = (1, 4))(x)
    x     = Conv2D(32,  kernel_size = (3, 3), padding='same', activation='relu')(x)
    x     = UpSampling2D(size = (1, 4))(x)
    x     = Conv2D(16,  kernel_size = (3, 3), padding='same', activation='relu')(x)
    x     = UpSampling2D(size = (1, 2))(x)
    x     = Conv2D(16,  kernel_size = (3, 3), padding='same', activation='relu')(x)
    x     = UpSampling2D(size = (1, 2))(x)
    x     = Conv2D(8,   kernel_size = (3, 3), padding='same', activation='relu')(x)
    x     = UpSampling2D(size = (1, 2))(x)
    x     = Conv2D(1,   kernel_size = (3, 3), padding='same', activation='tanh')(x)
    model = Model(inputs=[input], outputs=[x])
    return model
    

def discriminator(length, bins):
    '''
    Classify if a spectrogram is generated artificially or manually.

    length: Length of the spectrogram
    bins: Number of frequency bins
    '''    
    input = Input(shape = (length, bins, 1))
    x     = Conv2D(32, kernel_size = (3, 3), activation='relu')(input)
    x     = MaxPool2D((2,2))(x)
    x     = Conv2D(16, kernel_size = (3, 3), activation='relu')(x)
    x     = MaxPool2D((2,2))(x)
    x     = Conv2D(8, kernel_size = (3, 3), activation='relu')(x)
    x     = MaxPool2D((2,2))(x)
    x     = Flatten()(x)
    x     = Dense(1, activation='linear')(x)
    model = Model(inputs=[input], outputs=[x])
    return model


class TrainGAN:
    '''
    Train a Generative Adversarial Neural Network
    '''    

    def __init__(self, length):
        self.gen  = generator(length)
        self.disc = discriminator(length, 512)
        self.length = length
        self.adversarial()

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def adversarial(self):
        self.disc.compile(optimizer = RMSprop(lr=0.00005), loss = wasserstein_loss)
        self.set_trainable(self.disc, False)
        self.model_in  = Input(shape = (self.length))
        self.model_out = self.disc(self.gen(self.model_in))
        self.model     = Model(self.model_in, self.model_out) 
        self.model.compile(optimizer = RMSprop(lr=0.00005), loss = wasserstein_loss)
        self.set_trainable(self.disc, True)
        self.disc.trainable = True        
        self.disc.summary()
        self.model.summary()
        
    def one_step_generator(self, trace):
        '''
        One step of training for the generator.
        We pretend that fakes are real
        
        trace: a whistles trace
        '''
        assert(self.length == trace.length)
        y = np.ones((1,1))
        x = np.random.uniform(size = (2 * self.length, 1))
        x[0:self.length, :] = trace
        return self.model.train_on_batch(x, y)
    
    def one_step_disc(self, traces, reals, clip_threshold):
        valid = np.ones((1,1))
        fake = -np.ones((1,1))
        fakes = self.generator.predict(traces)
        
        
        
TrainGAN(100)
