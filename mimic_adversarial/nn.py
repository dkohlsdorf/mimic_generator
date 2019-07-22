import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import *


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


generator(100).summary()
discriminator(100, 512).summary()