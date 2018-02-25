from abc import ABCMeta, abstractmethod
from main.utils.sample_generator import SampleGenerator


class AlgoBase:
    __metaclass__ = ABCMeta
    columns = ["N", "Observations", "BoundType", "Unit"]

    def __init__(self, N, T, bound, statistic, confidence):
        """
        Initialize the different properties to create the bounds for sample distribution.
        Here, N= Sample Size, T= No. of Trails,
        bound= Type of Bound: Upper/Lower, statistic= mean/variance etc, confidence= value of confidence between 0-1
        """
        self.N = N 
        self.T = T 
        self.bound = bound 
        self.statistic = statistic
        self.confidence = confidence
        self.sample_generator = SampleGenerator()

    # TODO: add more abstract methods for computing variance etc.

    @abstractmethod
    def compute_mean(self):
        """
        Computes the upper and/or lower bound on the mean of samples
        :return: pandas DataFrame
        """
        pass
    
    # @abstractmethod
    def compute_variance(self):
        """
        Computes the upper and/or lower bound on the variance of samples
        :return: pandas DataFrame
        """
        pass

    def compute_statistic(self):
        if self.statistic == "mean":
            return self.compute_mean()
        if self.statistic == "variance":
            return self.compute_variance()
