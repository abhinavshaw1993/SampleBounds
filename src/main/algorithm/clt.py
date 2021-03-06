from algo_base import AlgoBase
from scipy.stats import t
import numpy as np
import pandas as pd


class CLT(AlgoBase):
    def compute_mean(self):

        boundsLimits = list()

        for N in xrange(1, self.N + 1):
            samples = self.sample_generator.generate_samples(N, self.T)

            for trials in xrange(self.T):
                est_mean = np.mean(samples[:, trials])
                est_std = np.sqrt(np.var(samples[:, trials]))
                est_mean_error = est_std / np.sqrt(N)

                df = N - 1  # Degrees of freedom for t distribution

                z_score = t.ppf(q=(self.confidence + (1 - self.confidence) / 2.0), df=df)

                if self.bound == "upper":
                    boundsLimits.append([N, est_mean + est_mean_error * z_score, 'Upper CLT', trials + 1])
                elif self.bound == "lower":
                    boundsLimits.append([N, est_mean - est_mean_error * z_score, 'Lower CLT', trials + 1])
                else:
                    boundsLimits.append([N, est_mean + est_mean_error * z_score, 'Upper CLT', trials + 1])
                    boundsLimits.append([N, est_mean - est_mean_error * z_score, 'Lower CLT', trials + 1])

        boundsDF = pd.DataFrame(data=boundsLimits, columns=["N", "Observations", "BoundType", "Unit"])
        return boundsDF
