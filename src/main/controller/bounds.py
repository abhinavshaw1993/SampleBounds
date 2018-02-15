from main.algorithm.clt import CLT


class BoundsExperiment:
    def __init__(self, n=10, t=10, bound="both", statistic="mean", algo=None, confidence=0.95):
        self.N = n
        self.T = t
        self.bound = bound
        self.statistic = statistic
        self.algo = ["CLT"] if algo is None else algo
        self.confidence = confidence

    def run_experiments(self):
        for algo in self.algo:
            if algo == "CLT":
                computer = CLT(self.N, self.T, self.bound, self.statistic, self.confidence)
                data = computer.compute_statistic()
                computer.plot_statistic(data)


if __name__ == "__main__":
    B = BoundsExperiment()
    B.run_experiments()
