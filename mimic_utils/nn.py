import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.constraints import *


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
    x     = Conv2D(32,  kernel_size = (3, 3), padding='same', activation='relu', kernel_constraint = WeightClip())(x)
    x     = UpSampling2D(size = (1, 4))(x)
    x     = Conv2D(32,  kernel_size = (3, 3), padding='same', activation='relu', kernel_constraint = WeightClip())(x)
    x     = UpSampling2D(size = (1, 4))(x)
    x     = Conv2D(16,  kernel_size = (3, 3), padding='same', activation='relu', kernel_constraint = WeightClip())(x)
    x     = UpSampling2D(size = (1, 2))(x)
    x     = Conv2D(16,  kernel_size = (3, 3), padding='same', activation='relu', kernel_constraint = WeightClip())(x)
    x     = UpSampling2D(size = (1, 2))(x)
    x     = Conv2D(8,   kernel_size = (3, 3), padding='same', activation='relu', kernel_constraint = WeightClip())(x)
    x     = UpSampling2D(size = (1, 2))(x)
    x     = Conv2D(1,   kernel_size = (3, 3), padding='same', activation='linear', kernel_constraint = WeightClip())(x)
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


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


class TrainGAN:
    '''
    Train a Generative Adversarial Neural Network
    '''    

    def __init__(self, length):
        self.gen  = generator(length)
        self.disc = discriminator(length, 512)
        self.length = length
        self.adversarial()

    def expand_trace(self, trace):
        expanded = np.random.uniform(size = (1, 2 * self.length))
        expanded[:, 0:self.length] = trace
        return expanded

    def create(self, trace):
        fake = self.gen.predict(self.expand_trace(trace))
        (_, t, d, _) = fake.shape
        return fake.reshape(t,d)
    
    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def adversarial(self):
        self.disc.compile(optimizer = RMSprop(lr=0.00005), loss = wasserstein_loss)
        self.set_trainable(self.disc, False)
        self.model_in  = Input(shape = (2 * self.length))
        self.model_out = self.disc(self.gen(self.model_in))
        self.model     = Model(self.model_in, self.model_out) 
        self.model.compile(optimizer = RMSprop(lr=0.00005), loss = wasserstein_loss)
        self.set_trainable(self.disc, True)
        self.disc.summary()
        self.model.summary()
        
    def one_step_disc(self, trace, real):        
        valid_label = np.ones((1,1))
        fake_label  = -np.ones((1,1))        
        fake        = self.gen.predict(self.expand_trace(trace))
                        
        d_loss_real = self.disc.train_on_batch(real, valid_label)
        d_loss_fake = self.disc.train_on_batch(fake, fake_label)
        d_loss      = 0.5 * (d_loss_real + d_loss_fake)

        return d_loss, d_loss_real, d_loss_fake
                
    def one_step_gen(self, trace):
        valid = np.ones((1,1))        
        return self.model.train_on_batch(self.expand_trace(trace), valid)      

    def train(self, traces, spectrograms, epochs):
        n = len(traces)
        for epoch in range(epochs):
            d_loss, d_loss_real, d_loss_fake = (0.0, 0.0, 0.0)
            for i in range(n):
                d, r, f = self.one_step_disc(
                    traces[i], spectrograms[i]
                )
                d_loss      += d
                d_loss_real += r
                d_loss_fake += f

            g_loss = 0.0
            for i in range(n):
               g = self.one_step_gen(traces[i])
               g_loss += g
            print("Discriminator [{} {} {}] || Generator {}".format(
                d_loss, d_loss_real, d_loss_fake, g_loss
            ))
