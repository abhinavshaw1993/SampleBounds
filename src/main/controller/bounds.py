from main.algorithm.clt import CLT
from main.algorithm.order_stats import ORDSTAT
from main.algorithm.chernoff_hoeffding import ChenoffHoeffding
from main.algorithm.massart import Massart
from main.algorithm.bootstrap import Bootstrap
from main.utils.plotting import plot_statistic
from main.utils.data_frame_processor import ProcessDataframe
import yaml
import pandas as pd


class BoundsExperiment:

    def __init__(self, n=None, t=None, bound=None, statistic=None, algo=None, confidence=None):

        # Reading Default Values form config file.
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        if not n:
            self.N = cfg["sample_statistics"]["N"]
        else:
            self.N = n

        if not t:
            self.T = cfg["sample_statistics"]["T"]
        else:
            self.T = t

        if not bound:
            self.bound = cfg["sample_statistics"]["bound"]
        else:
            self.bound = bound

        if not statistic:
            self.statistic = cfg["sample_statistics"]["statistic"]
        else:
            self.statistic = statistic

        if not confidence:
            self.confidence = cfg["sample_statistics"]["confidence"]
        else:
            self.confidence = confidence

        self.algo = ["CLT"] if algo is None else algo

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
        proc_df = ProcessDataframe(result)
        result = proc_df.process_dataframe(result)
        plot_statistic(result, N=self.N, T=self.T, statistic=self.statistic)


if __name__ == "__main__":
    B = BoundsExperiment(algo=["ORDSTAT", "CLT"])
    # B = BoundsExperiment(algo=["CLT"])
    # B = BoundsExperiment(algo=["ORDSTAT"])
    # B = BoundsExperiment(algo=["CHERNOFF-HOEFFDING"])
    # B = BoundsExperiment(algo=["MASSART"])
    # B = BoundsExperiment(algo=["BOOTSTRAP"])
    B.run_experiments()
