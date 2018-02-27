from main.algorithm.clt import CLT
from main.algorithm.order_stats import ORDSTAT
from main.algorithm.chernoff_hoeffding import ChenoffHoeffding
from main.algorithm.massart import Massart
from main.algorithm.bootstrap import Bootstrap
from main.utils.plotting import plot_statistic
import pandas as pd


class BoundsExperiment:
    def __init__(self, n=100, t=10, bound="both", statistic="mean", algo=None, confidence=0.95):

        self.N = n
        self.T = t
        self.bound = bound
        self.statistic = statistic
        self.algo = ["CLT"] if algo is None else algo
        self.confidence = confidence

    def run_experiments(self):
        result_df = []

        for algo in self.algo:
            print "Computing bounds on " + self.statistic + " using " + algo + " technique"

            if algo == "CLT":
                computer = CLT(self.N, self.T, self.bound, self.statistic, self.confidence)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "ORDSTAT":
                computer = ORDSTAT(self.N, self.T, self.bound, self.statistic, self.confidence)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "CHERNOFF-HOEFFDING":
                computer = ChenoffHoeffding(self.N, self.T, self.bound, self.statistic, self.confidence)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "MASSART":
                computer = Massart(self.N, self.T, self.bound, self.statistic, self.confidence)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "BOOTSTRAP":
                computer = Bootstrap(self.N, self.T, self.bound, self.statistic, self.confidence)
                data = computer.compute_statistic()
                result_df.append(data)

        result = pd.concat(result_df, ignore_index=True)
        plot_statistic(result, N=self.N, T=self.T, statistic=self.statistic)


if __name__ == "__main__":
    B = BoundsExperiment(algo=["ORDSTAT", "BOOTSTRAP"])
    # B = BoundsExperiment(algo=["CLT"])
    # B = BoundsExperiment(algo=["ORDSTAT"])
    # B = BoundsExperiment(algo=["CHERNOFF-HOEFFDING"])
    # B = BoundsExperiment(algo=["MASSART"])
    # B = BoundsExperiment(algo=["BOOTSTRAP"])
    B.run_experiments()
