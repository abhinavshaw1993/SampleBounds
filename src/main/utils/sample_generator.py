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

import numpy as np
from scipy.stats import truncnorm
import yaml


class SampleGenerator:
    # TODO: Add code/methods for generating distributions

    def __init__(self):
        """
        :param N: Number of samples
        :param T: Number of times to perform sampling
        """
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        self.left = cfg['sample_statistics']['left_support']
        self.right = cfg['sample_statistics']['right_support']
        self.mean = cfg['sample_statistics']['mean']
        self.std = cfg['sample_statistics']['stdev']
        self.distribution = cfg['sample_statistics']['distribution']
        self.random_seed = cfg['sample_statistics']['seed']
        # TODO: Add distribution relevant parameters like mean, std, alpha, beta etc.

    def generate_samples(self, N, T):
        if self.distribution == "normal":
            return self.normal(N, T)
        else:
            raise NotImplementedError

    def normal(self, N, T):
        """
        :return: samples from the normal distribution with support [self.left, self.right]
        """
        np.random.seed(self.random_seed)
        a, b = (self.left - self.mean) / self.std, (self.right - self.mean) / self.std
        samples = truncnorm.rvs(a=a, b=b, loc=self.mean, scale=self.std, size=(N, T))
        return samples
