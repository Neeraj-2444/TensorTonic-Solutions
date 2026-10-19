import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    n= len(A)
    m= len(A[0])
    At = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            At[i][j] = A[j][i]
    return At
    pass
