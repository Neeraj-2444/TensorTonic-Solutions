import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.array(y)
    if num_classes == None:
        num_classes = max(y)+1    
    one_hot = np.zeros((len(y),num_classes), dtype = float)
    for i in range(len(y)):
        for j in range(num_classes):
            if j == y[i]:
                one_hot[i][j] = 1
            else:
                one_hot[i][j] = 0
    return one_hot
        
    pass