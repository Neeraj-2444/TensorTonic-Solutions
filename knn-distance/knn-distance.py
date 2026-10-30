import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    diff = X_test[:, None, :] - X_train[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    idx = np.argsort(dist, axis=1)

    if k > X_train.shape[0]:
        padding = np.full((X_test.shape[0], k - X_train.shape[0]),-1)
        idx = np.concatenate((idx, padding), axis=1)

    return idx[:,:k].astype(int)
    pass