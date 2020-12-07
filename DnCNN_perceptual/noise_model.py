import numpy as np
import torch

def poissonpoissonnoise(X, min_eta, max_eta, Lambda, t):
    """
    Generate observation y (total SE counts) which obeys compound Poisson-Poisson distribution
    with total dose Lambda*t. The ground truth image is a 2-D array, scaled to [min_eta, max_eta]
    Args:
        X: ground truth image, i.e. mean SE yield.
        min_eta: minimum eta value after scaling and shifting.
        max_eta: maximum eta value after scaling and shifting.
        Lambda: dose rate. Dose for each sub-acquisition.
        t: dwell time.
    Returns:
        y: the total SE counts as an image. It's scaled in [0, 1].
        y_tr: a stack of t number of low-dose SE count images. It haven't been scaled yet.
    Shapes:
        input:
            X: [d1, d2], d1 and d2 are height and width of the image.
               or [c_chanel, d1, d2], where c_chanel is the number of color channel.
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

    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()

    if X.ndim == 2:
        # Get Dimensions of Image
        [d1, d2] = X.shape
        # Vectorize Image
        X = np.reshape(X, (d1*d2, 1))
    elif X.ndim == 3:
        [color_cn, d1, d2] = X.shape
        X = np.reshape(X, (color_cn, d1 * d2, 1))

    # Rescale X to be in [max_eta, min_eta]
    # slope = (max_eta - min_eta) / (X.max() - X.min())
    # distance = min_eta - X.min() * slope
    # eta = slope * X + distance
    eta = (max_eta - min_eta) * X + min_eta

    # Generate time-resolved ions
    ions = np.random.poisson(lam=Lambda, size=(*X.shape[:-1], t))

    # # M is the total ions.
    # M = np.sum(ions, axis=1)

    # Create time-resolved observations y_tr
    y_tr = np.random.poisson(eta * ions)

    # y is the total SE counts.
    y = np.sum(y_tr, axis=-1)

    # Rescale image so that it is \in [0,1]
    y = y / np.max(y)
    y = y.astype("float32")

    # Reshape for image
    y_tr = np.reshape(y_tr, newshape=(*X.shape[:-2], d1, d2, t))
    y = np.reshape(y, newshape=(*X.shape[:-2], d1, d2))

    return y, y_tr
