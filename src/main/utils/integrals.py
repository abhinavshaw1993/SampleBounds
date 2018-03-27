import numpy as np


def mean_integral(N, cdf_values, left=0.0, right=1.0):
    """
    Computes the mean from cdf of a distribution having positive finite support
    :param N: Number of samples
    :param left: Lower value of the support
    :param right: Upper value of the support
    :param cdf_values: Sorted values of the cdf of a distribution defined on the support [left, right]
    :return Mean value
    """
    dx = (right - left) / (N * 1.0)
    cdf_integral = 0.0

    for i in range(1, N):
        cdf_x0 = cdf_values[i - 1]
        cdf_x1 = cdf_values[i]
        cdf_integral += ((cdf_x0 + cdf_x1) / 2.0) * dx

    return right - cdf_integral


# def mean_integral_v2(cdf_values, ord_stats):
#     """
#     Computes the mean from cdf of a distribution having support [0, 1]
#     :param cdf_values: Sorted values of the cdf of a distribution defined on the support
#     :param ord_stats: Order statistics of the samples
#     :return:
#     """
#     cdf_values = np.hstack((0.0, cdf_values))
#     samples = np.hstack((0.0, ord_stats))
#     numerator = np.diff(cdf_values)
#     denominator = np.diff(samples)
#
#     pdf_values = numerator / denominator
#     mean = np.sum(np.multiply(ord_stats, pdf_values)) / np.sum(pdf_values)
#     return mean


def mean_integral_v2(cdf_values, ord_stats, bound_type='lower'):

    cdf_values = np.hstack((0.0, cdf_values, 1.0))
    samples = np.hstack((0.0, ord_stats, 1.0))
    Fx = cdf_values[:-1]

    if (bound_type == "lower"):
        Fx[0] = Fx[1]

    dx = np.diff(samples)
    mean = 1 - np.sum(np.multiply(Fx, dx))

    return mean