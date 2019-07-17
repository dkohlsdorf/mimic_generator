import numpy as np

ZERO = float('-inf')

def argmax(seq):
    '''
    Return the index and the value of the maximum element

    seq: some numeric sequence
    '''
    max_val = float('-inf')
    max_idx = 0
    for i, v in enumerate(seq):
        if v > max_val:
            max_val = v
            max_idx = i
    return max_idx, max_val


def entropy(p):
    '''
    Entropy of a discrete probability distribution

    p: discrete probability distribution
    '''
    return -np.sum(p * np.log(p + 1e-8))


def lgprob(x):
    if x == 0:
        return ZERO
    else:
        return np.log(x)
