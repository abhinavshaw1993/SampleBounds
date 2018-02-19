from main.algorithm.clt import CLT
from main.algorithm.order_stats import ORDSTAT
from main.algorithm.chernoff_hoeffding import ChenoffHoeffding
from main.utils.plotting import plot_statistic
from main.utils.sample_generator import SampleGenerator
import yaml


class BoundsExperiment:
    def __init__(self, n=100, t=10, bound="both", statistic="mean", algo=None, confidence=0.95):

        self.N = n
        self.T = t
        self.bound = bound
        self.statistic = statistic
        self.algo = ["CLT"] if algo is None else algo
        self.confidence = confidence
        self.sample_generator = SampleGenerator(self.N, self.T)

    def run_experiments(self):
        result_df = None

        samples = self.sample_generator.normal()

        for algo in self.algo:
            if algo == "CLT":
                computer = CLT(self.N, self.T, self.bound, self.statistic, self.confidence, samples)
                data = computer.compute_statistic()
                if result_df is None:
                    result_df = data
                else:
                    result_df.append(data, ignore_index=True)

            if algo == "ORDSTAT":
                computer = ORDSTAT(self.N, self.T, self.bound, self.statistic, self.confidence, samples)
                data = computer.compute_statistic()
                if result_df is None:
                    result_df = data
                else:
                    result_df.append(data, ignore_index=True)

            if algo == "CHERNOFF-HOEFFDING":
                computer = ChenoffHoeffding(self.N, self.T, self.bound, self.statistic, self.confidence, samples)
                data = computer.compute_statistic()
                if result_df is None:
                    result_df = data
                else:
                    result_df.append(data, ignore_index=True)
        plot_statistic(result_df, N=self.N, T=self.T)

if __name__ == "__main__":
    # B = BoundsExperiment(algo=["CLT", "ORDSTAT"])
    B = BoundsExperiment(algo=["CHERNOFF-HOEFFDING"])
    B.run_experiments()
