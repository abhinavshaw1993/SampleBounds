from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AlgoBase:
    __metaclass__ = ABCMeta

    def __init__(self, N, T, bound, statistic, confidence, samples):
        self.N = N
        self.T = T
        self.bound = bound
        self.statistic = statistic
        self.confidence = confidence
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
