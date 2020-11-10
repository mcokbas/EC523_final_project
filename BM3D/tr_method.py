import numpy as np
from scipy.stats import poisson
from timeit import default_timer as timer

def neyman_log_loss(y, eta, lam_single):
    """
    Compute the log-likelihood of y which follows Neyman Type-A PMF.
    The observation y is generated with total dose lam_single * y.shape[1].

    Args:
        y: (ndarray) total SE counts.
        eta: (ndarray) mean SE yield.
        lam_single (float): dose for each sub-acquisition.

    Returns:
        log_pmf_y (ndarray): log-likelihood of observing y.

    Shapes:
        Inputs:
            y: (nn, dd, 1) where nn represents the number of pixels (length x width)
               of the ground truth image. dd is the number of sub-acquisitions.
            eta: (nn,)
            lambda_single: scalar.

        Output:
            log_pmf_y: (nn,)
    """
    m = np.arange(15).reshape((1, 1, -1))
    eta = eta.reshape((-1, 1, 1))
    y = y[:, :, np.newaxis]

    pmf_m = poisson.pmf(m, lam_single)
    pmf_y_given_m = poisson.pmf(y, eta * m)
    log_pmf_y = np.sum(np.log(np.sum(pmf_m * pmf_y_given_m, axis=2)), axis=1)

    return log_pmf_y


def trml_estimate(y, lam_single):
    """
    Computes the estimated eta value from using Expectation Maximization (EM) algorithm.
    It will terminate after 5000 steps or the absolute difference of two consecutive estimated etas
    is less than tolerance.

    Args:
        y (ndarray): total SE counts.
        lam_single (float): dose for each sub-acquisition.

    Returns:
        eta (ndarray): estimated eta.

    Shapes:
        Inputs
            y: (nn, dd), where nn is the number of pixels. dd is the number of sub-acquisitions.
            lam_single: scalar.
        Output:
            eta: (nn,)
    """
    m = np.arange(15).reshape((1, 1, -1))
    pmf_m = poisson.pmf(m, lam_single)
    y = y[:, :, np.newaxis]
    nn, dd, _ = y.shape

    # Initialize the prior probability w_m^{(i)} = P(M=m|Y_i=y_i).
    # Firstly initiate eta which is the conventional estimator.
    eta = y.sum(axis=1) / lam_single / dd
    eta = eta.reshape((-1, 1, 1))

    steps = 5000
    tol = 1e-5
    t_start = timer()
    eta_hat = []
    for i in range(y.shape[0]):
        y_vec = y[i, :, :]
        eta_vec = eta[i, :, :]

        eta_store = []
        for step in range(steps):
            # E-step.
            # Calculate the prior probability w_m^{(i)} = P(M=m|Y_i=y_i).
            pmf_y_given_m = poisson.pmf(y_vec, eta_vec * m)
            pmf_y_and_m = pmf_y_given_m * pmf_m
            pmf_y_and_m_sum = np.sum(pmf_y_and_m, axis=2, keepdims=True)

            w_m = pmf_y_and_m / pmf_y_and_m_sum

            # M-step.
            # Update eta.
            y_wm_sum = np.sum(y_vec * w_m)
            m_wm_sum = np.sum(m * w_m)

            eta_new = y_wm_sum / m_wm_sum
            eta_vec.fill(eta_new)

            if step > 0 and np.abs(eta_new - eta_store[-1]) <= tol:
                eta_hat.append(eta_new)
                # print("Pixel {}, elapsed time = {:.2f} seconds".format(i, timer() - t_start))
                break
            eta_store.append(eta_new)

    print("TRML method takes {:.2f} seconds".format(timer() - t_start))

    return np.array(eta_hat)



