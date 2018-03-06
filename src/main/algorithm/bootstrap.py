from algo_base import AlgoBase
import numpy as np
import scipy as sc
import pandas as pd


class Bootstrap(AlgoBase):
    """
    Class for Bootstrap Bounds
    The Confidence is set to 95* for this bound. Might mae it configurable in the future.
    """

    def compute_mean(self):
        """
        Computes mean with the DELTA method of Bootstrapping.
        For Bootstrap T(Trails) is equivalent to number Bootstrap sampling experiment.
        """
        bootstrap_exp_count = 1000
        resample_count = 1000
        bound_limits = list()

        for N in xrange(2, self.N + 1):
            samples = self.sample_generator.generate_samples(N, self.T)

            for T in range(self.T):
                m_l, m_u = self.compute_boostrap_experiment(N, samples[:, T], resample_count, bootstrap_exp_count)

                if self.bound == "lower":
                    bound_limits.append([N, m_l, 'Lower Bootstrap', T + 1])
                elif self.bound == "upper":
                    bound_limits.append([N, m_u, 'Upper Bootstrap', T + 1])
                else:
                    bound_limits.append([N, m_l, 'Lower Bootstrap', T + 1])
                    bound_limits.append([N, m_u, 'Upper Bootstrap', T + 1])

        return pd.DataFrame(data=bound_limits, columns=Bootstrap.columns)

    def compute_boostrap_experiment(self, N, samples, resample_count, bootstrap_exp_count):
        """
        Performs one Bootstrap experiment.
        :param N: Number of samples, for optimization Purposes.
        :param samples: Samples for which you want to Bootstrap.
        :param resample_count: Number of samples you want while re-sampling from the original samples.
        :param bootstrap_exp_count: Number of times you want to re-sample from the original samples.
        :return: Delta Star.
        """
        sample_mean = np.mean(samples)
        delta_star = list()
        for i in range(bootstrap_exp_count):
            bootstrap_samples = samples[np.random.randint(0, N, resample_count)]
            x_bar_star = np.mean(bootstrap_samples)
            delta_star.append(x_bar_star - sample_mean)

        # for 95% confidence.
        delta = sc.stats.mstats.mquantiles(delta_star, (0.05, 0.95))

        return sample_mean - delta[1], sample_mean - delta[0]
