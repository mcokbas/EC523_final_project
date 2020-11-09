import numpy as np

def poissonpoissonnoise(X, min_eta, max_eta , Lambda, t):
    """
    Generate observation y (total SE counts) which obeys compound Poisson-Poisson distribution
    with total dose Lambda*t. The ground truth image is a 2-D array, scaled to [min_eta, max_eta]

    Args:
        X: ground truth image, i.e. mean SE yield.
        min_eta: minimum eta value after scaling and shifting.
        max_eta: maximum eta value after scaling and shifting.
        Lambda: dose rate.
        t: dwell time.

    Returns:
        y: the total SE counts as an image. It's scaled in [0, 1].
        y_tr: a stack of t number of low-dose SE count images. It haven't been scaled yet.

    Shapes:
        input:
            X: [d1, d2], d1 and d2 are height and width of the image.
            min_eta: scalar.
            max_eta: scalar.
            Lambda: scalar
            t: scalar.
        output:
            y: [d1, d2]
            y_tr: [d1, d2, t]

    """
    #X: Input matrix containing grayscale values as double [0,1]
    #min_eta: Minimum eta after scaling and shifting
    #max_eta: Maximum eta after scaling and shifting
    #Lambda: Poisson process rate per unit time
    #t: Dwell time

    # Get Dimensions of Image
    [d1, d2] = X.shape
    # Vectorize Image
    X = np.reshape(X, [d1*d2, 1])

    #Means for # of SE's (M)
    eta = (max_eta - min_eta)*X + min_eta

    # Generate time-resolved ions
    ions = np.random.poisson(lam=Lambda, size=(X.size, t))

    # M is the total ions.
    M = np.sum(ions, axis=1)

    # Create time-resolved observations y_tr
    y_tr = np.random.poisson(eta * ions)

    # y is the total SE counts.
    y = np.sum(y_tr, axis=1)

    #Rescale image so that it is \in [0,1]
    y = y / np.max(y)

    #Reshape for image
    y_tr = y_tr.reshape([d1, d2, t])
    y = np.reshape(y, [d1, d2])

    return y, y_tr

