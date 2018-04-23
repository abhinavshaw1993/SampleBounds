from main.algorithm.clt import CLT
from main.algorithm.order_stats import ORDSTAT
from main.algorithm.chernoff_hoeffding import ChenoffHoeffding
from main.algorithm.massart import Massart
from main.algorithm.bootstrap import Bootstrap
from main.utils.plotting import plot_statistic
from main.utils.data_frame_processor import ProcessDataframe
from main.utils.sample_generator import SampleGenerator
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

        # Initializing Parameters for the sample generator.
        # The mean and stdev can only be set for the trunc Norm.

        self.left = cfg['sample_statistics']['left_support']
        self.right = cfg['sample_statistics']['right_support']
        self.mean = cfg['sample_statistics']['mean']
        self.stdev = cfg['sample_statistics']['stdev']
        self.distribution = cfg['sample_statistics']['distribution']
        self.random_seed = cfg['sample_statistics']['seed']

        # Initializing Sample Generator
        self.sg = SampleGenerator(self.left, self.right, self.mean, self.stdev, self.distribution, self.random_seed)
        if self.statistic == "mean":
            self.true_mean = self.sg.true_mean()
            self.true_variance = None
        if self.statistic == "variance":
            self.true_variance = self.sg.true_variance()
            self.true_mean = None

    def run_experiments(self):
        result_df = []

        for algo in self.algo:
            print "Computing bounds on " + self.statistic + " using " + algo + " technique"

            if algo == "CLT":
                computer = CLT(self.N, self.T, self.bound, self.statistic, self.confidence, self.sg)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "ORDSTAT":
                computer = ORDSTAT(self.N, self.T, self.bound, self.statistic, self.confidence, self.sg)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "CHERNOFF-HOEFFDING":
                computer = ChenoffHoeffding(self.N, self.T, self.bound, self.statistic, self.confidence, self.sg)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "MASSART":
                computer = Massart(self.N, self.T, self.bound, self.statistic, self.confidence, self.sg)
                data = computer.compute_statistic()
                result_df.append(data)

            if algo == "BOOTSTRAP":
                computer = Bootstrap(self.N, self.T, self.bound, self.statistic, self.confidence, self.sg)
                data = computer.compute_statistic()
                result_df.append(data)

        result = pd.concat(result_df, ignore_index=True)
        print ("obtained result")
        # Creating instance of DataProcessor.
        proc_df = ProcessDataframe(result)
        print ("data frame processed")
        result = proc_df.process_dataframe(result)
        print ("going for plotting")
        # Plotting Statistics.
        plot_statistic(result, N=self.N, T=self.T, true_mean=self.true_mean, true_variance=self.true_variance , statistic=self.statistic)


if __name__ == "__main__":
    #B = BoundsExperiment(algo=["ORDSTAT","CLT","CHERNOFF_HOEFFDING","MASSART"])
    # B = BoundsExperiment(algo=["CLT"])
    B = BoundsExperiment(algo=["ORDSTAT"])
    # B = BoundsExperiment(algo=["CHERNOFF-HOEFFDING"])
    # B = BoundsExperiment(algo=["MASSART"])
    # B = BoundsExperiment(algo=["BOOTSTRAP"])
    B.run_experiments()
