import math
import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    classes = np.unique(y_train)
    predictions = []

    for x in X_test:
        scores = []
        for c in classes:
            X_c = X_train[y_train == c]
            prior = len(X_c) / len(X_train)
            mean = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0) + 1e-9

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) +
                ((x - mean) ** 2) / var)
            
            score = np.log(prior) + log_likelihood
            scores.append(score)

        predictions.append(classes[np.argmax(scores)])

    return predictions
    pass