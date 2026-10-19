import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    s=0.0
    y = np.array(y)
    for i in set(y):
        pi = np.count_nonzero(y==i)/len(y)
        s += pi * np.log2(pi)
    return -s
    pass