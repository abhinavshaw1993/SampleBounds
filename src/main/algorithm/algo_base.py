from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AlgoBase:
    __metaclass__ = ABCMeta

    def __init__(self, N, T, bound, statistic, confidence):
        self.N = N
        self.T = T
        self.bound = bound
        self.statistic = statistic
        self.confidence = confidence

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

    @staticmethod
    def plot_statistic(df):
        """
        :param df: pandas DataFrame containing upper/lower bounds for a statistic (mean, variance etc.)
        :return: matplotlib axis
        """

        try:
            assert isinstance(df, pd.DataFrame)
            assert df.columns == ["N", "Observations", "BoundType", "Unit"]
        except Exception as e:
            # print "error({0}): {1}".format(e.errno, e.strerror)
            print e

        plt.subplot()
        sns.tsplot(data=df, time="N", value="Observations", condition="BoundType", unit="Unit", ci=100)
        plt.show()
