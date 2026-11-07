import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    
    X = np.array(X, dtype=float)

    if X.ndim == 1:
        valid = X[~np.isnan(X)]

        if len(valid) == 0:
            fill_value = 0
        elif strategy == "mean":
            fill_value = np.mean(valid)
        elif strategy == "median":
            fill_value = np.median(valid)
        else:
            raise ValueError("Invalid strategy")

        X[np.isnan(X)] = fill_value
        return X

    for j in range(X.shape[1]):
        col = X[:, j]
        valid = col[~np.isnan(col)]

        if len(valid) == 0:
            fill_value = 0
        elif strategy == "mean":
            fill_value = np.mean(valid)
        elif strategy == "median":
            fill_value = np.median(valid)
        else:
            raise ValueError("Invalid strategy")

        col[np.isnan(col)] = fill_value

    return X
    pass