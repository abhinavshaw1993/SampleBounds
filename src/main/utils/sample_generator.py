"""
This module is responsible for generating samples from a fixed set of distributions
All distributions have a fixed positive support [left, right] such that 0<=left<=right
Samples from the following distributions can be generated
 - Normal distribution
 - Beta distribution
 - Exponential
 - Double Exponential
 - Mixture of gaussians
"""

from scipy.stats import truncnorm
import yaml

class SampleGenerator:
    # TODO: Add code/methods for generating distributions

    def __init__(self, N, T):
        """
        :param N: Number of samples
        :param T: Number of times to perform sampling
        """
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        self.N = N
        self.T = T
        self.left = cfg['sample_statistics']['left_support']
        self.right = cfg['sample_statistics']['right_support']
        self.mean = cfg['sample_statistics']['mean']
        self.std = cfg['sample_statistics']['stdev']
        # TODO: Add distribution relevant parameters like mean, std, alpha, beta etc.

    def normal(self):
        """
        :return: samples from the normal distribution with support [a, b]
        """
        a, b = (self.left - self.mean) / self.std, (self.right - self.mean) / self.std
        samples = truncnorm.rvs(a=a, b=b, loc=self.mean, scale=self.std, size=(self.T, self.N))
        return samples
