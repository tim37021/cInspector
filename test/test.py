import cInspector
import numpy as np

def auto_correlation(data, min_lag, max_lag, window_size):
    """
        Auto correlation
        auto correlation with pure numpy approach
    """
    assert len(data) >= window_size, 'window size cannot be larger than input size'
    
    min_lag = min(len(data) - window_size, min_lag)
    max_lag = min(len(data) - window_size, max_lag)

    corr = np.zeros(max_lag - min_lag + 1, dtype=np.float32)
    for x in range(min_lag, max_lag+1):
        corr[x-min_lag] = np.sum(np.abs(data[-window_size:] - data[-window_size-x: -x if x > 0 else None]) / window_size)
    return corr


def test_auto_correlation():
    t = np.linspace(0, 1, 44100, dtype=np.float32)

    x = np.sin(2*np.pi*441*t)[:1024]
    a = auto_correlation(x, 32, 500, 1024)
    b = cInspector.auto_correlation(x, 32, 500, 1024)
    assert ((a-b)<1e-4).all(), 'Basic test failed'

    x = np.sin(2*np.pi*441*t)[:2048:2]
    a = auto_correlation(x, 32, 500, 500)
    b = cInspector.auto_correlation(x, 32, 500, 500)
    assert ((a-b)<1e-4).all(), 'Postive stride test failed'

    # inverse
    x = np.sin(2*np.pi*441*t)[:1024:-1]
    a = auto_correlation(x, 32, 500, 500)
    b = cInspector.auto_correlation(x, 32, 500, 500)
    assert ((a-b)<1e-4).all(), 'Negative stride test failed'

def test_hcPeakValley():
    pv = cInspector.hcPeakValley()
    t = np.linspace(0, 1, 44100, dtype=np.float32)
    x = np.sin(2*np.pi*441*t)[:2048]
    p, v = pv(x)
    
    
    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.scatter(p, x[p])
    plt.scatter(v, x[v])

    plt.show()

    assert (np.cos(2*np.pi*441*t[p]) < 1e-4).all(), 'Peak Slope != 0'

if __name__ == '__main__':
    test_hcPeakValley()