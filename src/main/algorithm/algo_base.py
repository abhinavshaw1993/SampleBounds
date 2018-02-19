from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AlgoBase:
    __metaclass__ = ABCMeta

    def __init__(self, N, T, bound, statistic, confidence, samples):
        self.N = N # Sample Size
        self.T = T # No of trials
        self.bound = bound # Type of Bound: Upper/Lower
        self.statistic = statistic # what kind of statistical property eg mean, variance etc.
        self.confidence = confidence # value of confidence 
        self.samples = samples

    # TODO: add more abstract methods for computing variance etc.

    @abstractmethod
    def compute_mean(self):
        """
        Computes the upper and/or lower bound on the mean of samples
        :return: pandas DataFrame
        """
        pass

    def compute_statistic(self):
        if self.statistic == "mean":
            return self.compute_mean()
