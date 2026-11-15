import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.array(x,dtype = float)
    sample_mean = np.mean(x)
    s = np.sqrt((np.sum((x-sample_mean)**2))/(len(x)-1))
    if s == 0:
        if sample_mean == mu0:
            return 0.0
        else:
            return float('inf')

    t = ((sample_mean - mu0)* np.sqrt(len(x))/s )

    return float(t)
    
    pass