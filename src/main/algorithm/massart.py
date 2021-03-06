from algo_base import AlgoBase
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import pandas as pd
import yaml

class Massart(AlgoBase):
    """
    Class for Massart Bounds Bounds.
    The computation is only possible for 3 or more samples.
    :return: Return DataFrame used for Time Series plot Seaborn.
    """
    def compute_cdf(self, samples):

        # Extracting N from
        N, T = samples.shape
        e = self.compute_epsilon(N)
        linspace = np.linspace(0, 1, N)
        upper_envelope = lower_envelope = None
        F_x = np.zeros(samples.shape)

        for i in range(T):
            ecdf = ECDF(samples[:, i])
            F_x[:, i] = ecdf(linspace)

        if self.bound == "upper":
            upper_envelope = F_x + e
        elif self.bound == "lower":
            lower_envelope = F_x - e
        else:
            upper_envelope = F_x + e
            lower_envelope = F_x - e

        return upper_envelope, lower_envelope

    def compute_mean(self):
        bound_limits = list()

        # Computation start from 3 and above since there is no point calculating
        #  Massart for very low values of N.
        for N in range(1, self.N + 1):
            e = self.compute_epsilon(N)
            samples = self.sample_generator.generate_samples(N, self.T)

            for T in range(self.T):
                m_l, m_u = self.estimate_bound(N, samples[:, T], e)

                if self.bound == "lower":
                    bound_limits.append([N, m_l, 'Lower Massart', T + 1])
                elif self.bound == "upper":
                    bound_limits.append([N, m_u, 'Upper Massart', T + 1])
                else:
                    bound_limits.append([N, m_l, 'Lower Massart', T + 1])
                    bound_limits.append([N, m_u, 'Upper Massart', T + 1])

        return pd.DataFrame(data=bound_limits, columns=Massart.columns)

    def compute_epsilon(self, N):
        """
        Computes epsilon (Bound parameter for Chernoff-Hoeffding bounds.
        :return: Returns epsilon.
        """
        intermediate = np.log((1 - self.confidence) / 2.0) / (-2.0 * N)
        return np.sqrt(intermediate)

    def estimate_bound(self, N, samples, e):
        """
        Computes Upper and Lower Bounds on Mean as per algorithm by Learned-Miller.
        :param N: Number of samples.
        :param samples: Samples on which the bounds are to be given.
        :param e: Epsilon calculated by confidence interval alopha.
        :return: Returns Lower and Upper Bounds.
        """
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        a = cfg['sample_statistics']['left_support']
        b = cfg['sample_statistics']['right_support']
        N = float(N)
        order_stats = np.sort(samples)
        m_l = a * e + order_stats[0] * (1.0 / N)
        m_u = order_stats[0] * max(0, (1.0 / N) - e)

        for i in range(1, int(N)):
            m_l += (min(1, ((i+1) / N) + e) - min(1, ((i+1) - 1) / N + e)) * order_stats[i]
            m_u += (max(0, ((i+1) / N) - e) - max(0, ((i+1) - 1) / N - e)) * order_stats[i]

        m_u += e * b

        return m_l, m_u
