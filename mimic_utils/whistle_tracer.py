import numpy as np
from mimic_utils.numerics import * 


def trace(whistle, radius, smooth_size):
    '''
    Traces a whistle in a spectrogram using a probabilistic discrete filter

    whistle: spectrogram of a whistle
    radius: check previous bins backwards in radius
    smooth_size: window for moving average to smooth scoring function
    '''
    w, h = whistle.shape
    h = int(h/2)
    
    # Convert whistle to probability distribution
    whistle   = np.abs(whistle[:, 0:h])
    whistle  /= np.max(whistle)
    
    # Initialize Tracer
    dp = np.ones((w, h)) * float('-inf')
    bp = np.zeros((w,h), dtype=np.int)
    for f in range(h):
        dp[0, f] = whistle[0, f]

    # Viterbi style tracing of the whistle
    for t in range(1, w):
        for f in range(0, h):
            start = int(max(f - radius, 0))
            stop  = int(min(f + radius , h))
            bp[t,f], dp[t, f] = argmax([
                dp[t - 1, prev] + lgprob(whistle[t - 1, prev]) for prev in range(start, stop)
            ])
            bp[t, f] += start
            dp[t, f] += lgprob(whistle[t, f])
    
    # Compute the path from the viterbi matrix 
    # For each timestep also compute the entropy of the spectrogram slice
    path   = np.zeros(w, dtype=np.int)
    scores = np.zeros(w)
    (path[w - 1], scores[w - 1]) = argmax([dp[w - 1, f] for f in range(0, h)])            
    i = w - 2
    while i > 0:
        path[i]   = bp[i, path[i + 1]]
        scores[i] = entropy(whistle[i, :])
        i -= 1

    # Smooth the scores using a moving average 
    overlap   = int(smooth_size / 2)
    scores    = np.convolve(scores, np.ones((smooth_size,))/smooth_size, mode='full')
    threshold = np.median(scores[overlap:-overlap])
    for i in range(overlap, len(scores) - overlap):
        if scores[i] <= threshold: 
            path[i - overlap] = 0.0    
    return path, scores
