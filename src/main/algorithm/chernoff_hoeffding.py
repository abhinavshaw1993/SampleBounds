from algo_base import AlgoBase
import numpy as np
import pandas as pd


class ChenoffHoeffding(AlgoBase):
    """
    Class for Chernoff-Hoeffding Bounds
    """

    def compute_mean(self):

        bound_limits = list()

        for N in xrange(1, self.N + 1):
            e = self.compute_epsilon(N)
            samples = self.sample_generator.generate_samples(N, self.T)

            for T in xrange(self.T):
                mean = np.mean(samples[:, T])

                if self.bound == "upper":
                    bound_limits.append([N, mean + e, 'Upper CH', T + 1])
                elif self.bound == "lower":
                    bound_limits.append([N, mean - e, 'Lower CH', T + 1])
                else:
                    bound_limits.append([N, mean + e, 'Upper CH', T + 1])
                    bound_limits.append([N, mean - e, 'Lower CH', T + 1])

        bounds_df = pd.DataFrame(data=bound_limits, columns=ChenoffHoeffding.columns)
        return bounds_df

    def compute_epsilon(self, N):
        """
        Computes epsilon (Bound parameter for Chernoff-Hoeffding bounds.
        :return: Returns epsilon.
        """
        intermediate = - np.log((1 - self.confidence) / 2.0) / (2.0 * N)
        return np.sqrt(intermediate)
