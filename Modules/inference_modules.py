import numpy as np


def bootstrap_measure(objects, measuring_function, n_samples: int = 1000):
    """Calculate the bootstrap mean and standard deviation.

    Parameters
    ----------
    objects : sequence
        Data points to sample with replacement.
    measuring_function : callable
        Function applied to each bootstrap sample.
    n_samples : int, optional
        Number of bootstrap samples to generate. Defaults to ``1000``.

    Returns
    -------
    tuple
        ``(mean, std)`` of ``measuring_function`` evaluated on bootstrap samples.
    """

    n = len(objects)
    measuring_values = [
        measuring_function(np.random.choice(objects, size=n, replace=True))
        for _ in range(n_samples)
    ]
    return float(np.mean(measuring_values)), float(np.std(measuring_values))
