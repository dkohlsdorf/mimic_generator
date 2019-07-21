
import random
import numpy as np

from collections import namedtuple


class SampleRange(namedtuple('SampleRange', 'start stop')):

    def sample(self):
        if isinstance(self.start, int):
            return random.randint(self.start, self.stop)
        else:
            return np.random.uniform(self.start, self.stop)


# Dynamic Parameters
HARMONICS = SampleRange(1,   8)
MARGIN    = SampleRange(3,   8)
SHIFT     = SampleRange(1,   5)  

# Fixed Params
FFT_WIN    = 1024
FFT_STEP   = 512
TRACE_RAD  = 10
SMOOTH_ENT = 64
LOUDNESS   = 1.0
