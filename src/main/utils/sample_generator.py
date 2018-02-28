"""
This module is responsible for generating samples from a fixed set of distributions
All distributions have a fixed positive support [left, right] such that 0<=left<=right
Samples from the following distributions can be generated
 - Normal distribution
 - Beta distribution
 - Exponential

The module will no start returning the mean as well, since it is hard to control the mean

"""

import numpy as np
from scipy.stats import truncnorm
from scipy.stats import expon
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
        self.distribution = cfg['sample_statistics']['distribution']
        self.random_seed = cfg['sample_statistics']['seed']
        # TODO: Add distribution relevant parameters like mean, std, alpha, beta etc.

    def generate_samples(self, N, T):
        if self.distribution == "normal":
            return self.normal(N, T)
        elif self.distribution == "uniform":
            return self.uniform(N, T)
        elif self.distribution == "exponential":
            return self.exponential(N, T)
        else:
            raise NotImplementedError

    def normal(self, N, T):
        """
        :return: samples from the normal distribution with support [self.left, self.right]
        """
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        self.mean = cfg['normal']['true_mean']
        self.std = cfg['normal']['scale']
        np.random.seed(self.random_seed)
        a, b = (self.left - self.mean) / self.std, (self.right - self.mean) / self.std
        samples = truncnorm.rvs(a=a, b=b, loc=self.mean, scale=self.std, size=(N, T))
        return samples

    def uniform(self, N, T):
        """
        :return: samples from the uniform distribution with support [self.left, self.right]
        """
        a = self.left
        b = self.right
        np.random.seed(self.random_seed)
        return np.random.uniform(a, b, size=(N, T))

    def exponential(self, N, T):
        """
        :return: samples from the uniform distribution with support [self.left, self.right]
        Notes - We'll be using exponential with a support 0, 1. To achieve this we'll be
        truncating the exponential distribution after 1. The support of the exponential
        distribution is from 0 to infinity by default. We will be keeping the scale low so that
        not a lot of samples are truncated from the distribution and we have some consistency.
        """
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        loc = cfg["exponential"]["loc"]
        scale = cfg["exponential"]["scale"]
        np.random.seed(self.random_seed)
        samples = expon.rvs(loc=loc, scale=scale, size=(N, T))
        # print(np.mean(samples[:, 1]))
        return samples
