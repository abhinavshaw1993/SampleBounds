def mean_integral(N, left, right, cdf_values):
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
        x0 = left + (i - 1) * dx
        x1 = left + i * dx
        cdf_x0 = cdf_values[i - 1]
        cdf_x1 = cdf_values[i]
        cdf_integral += ((cdf_x0 + cdf_x1) / 2.0) * dx

    return right - cdf_integral
