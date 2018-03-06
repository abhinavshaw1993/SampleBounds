"""
This module is responsible for generating samples from a fixed set of distributions
All distributions have a fixed positive support [left, right] such that 0<=left<=right
Samples from the following distributions can be generated
 - Normal distribution
 - Beta distribution
 - Exponential distribution
 - Uniform distribution

The module will no start returning the mean as well, since it is hard to control the mean

"""

from scipy.stats import truncnorm
from scipy.stats import truncexpon
from scipy.stats import beta
from scipy.stats import uniform


class SampleGenerator:
    # TODO: Add code/methods for generating distributions

    def __init__(self, left, right, mean, stdev, distribution, random_seed):
        """
        :param N: Number of samples
        :param T: Number of times to perform sampling
        :param mean: mean of the samples, used for truncated normal distribution
        :param stdev: Standard Deviation, used for truncated normal distribution
        """

        self.left = left
        self.right = right
        self.distribution = distribution
        self.random_seed = random_seed
        self.mean = mean
        self.stdev = stdev
        # TODO: Add distribution relevant parameters like mean, std, alpha, beta etc.

    def generate_samples(self, N, T):
        if self.distribution == "normal":
            return self.normal(N, T)
        elif self.distribution == "uniform":
            return self.uniform(N, T)
        elif self.distribution == "exponential":
            return self.exponential(N, T)
        elif self.distribution == "beta":
            return self.beta(N, T)
        else:
            raise NotImplementedError

    def normal(self, N, T):
        """
        :return: samples from the normal distribution with support [self.left, self.right]
        """

        a, b = (self.left - self.mean) / self.stdev, (self.right - self.mean) / self.stdev
        samples = truncnorm.rvs(a=a, b=b, loc=self.mean, scale=self.stdev, size=(N, T), random_state=self.random_seed)
        return samples

    def uniform(self, N, T):
        """
        :return: samples from the uniform distribution with support [self.left, self.right]
        """

        a = self.left
        b = self.right
        samples = uniform.rvs(loc=self.left, scale=self.right, size=(N,T), random_state=self.random_seed)

        return samples

    def exponential(self, N, T):
        """
        :return: samples from the exponential distribution with support [self.left, self.right]
        Notes - We'll be using exponential with a support 0, 1. To achieve this we'll be
        truncating the exponential distribution after 1. The support of the exponential
        distribution is from 0 to infinity by default. We will be keeping the scale low so that
        not a lot of samples are truncated from the distribution and we have some consistency.
        """

        # truncexpon moves from 0 to b.
        return truncexpon.rvs(b=self.right, loc=self.mean, scale=self.stdev, size=(N, T), random_state=self.random_seed)

    def beta(self, N, T):
        """
        :return: samples from the beta distribution with with the following parameters
        a=0.5, b=1, loc=left support, scale=right support (Already tested with these params, they give us a beta with 0,1 support)
        """
        return beta.rvs(a=0.5, b=1, loc=self.left, scale=self.right, size=(N, T), random_state=self.random_seed)

    def true_mean(self):
        def true_mean_normal():
            a, b = (self.left - self.mean) / self.stdev, (self.right - self.mean) / self.stdev
            return truncnorm.mean(a, b, loc=self.mean, scale=self.stdev)

        def true_mean_uniform():
            return (self.right - self.left) / 2.0

        def true_mean_exponential():
            return truncexpon.mean(b=self.right, loc=self.mean, scale=self.stdev)

        def true_mean_beta():
            return beta.mean(a=0.5, b=1, loc=self.left, scale=self.right)

        compute_true_mean = {
                   "normal": true_mean_normal,
                   "uniform": true_mean_uniform,
                   "exponential": true_mean_exponential,
                   "beta": true_mean_beta
                   }

        return compute_true_mean[self.distribution]()
