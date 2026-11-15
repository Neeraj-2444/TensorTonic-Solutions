import numpy as np

def chi2_independence(C: list) -> dict:
    """
    Returns a dictionary with chi2 and expected.
    """
    C = np.array(C, dtype=float)
    row_sum = np.sum(C, axis=1)
    col_sum = np.sum(C, axis=0)

    total = np.sum(C)

    expected = np.outer(row_sum, col_sum) / total

    chi2 = np.sum((C - expected)**2 / expected).item()

    return {
        "chi2": float(chi2) , "expected": expected
    }

    pass