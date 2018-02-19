from algo_base import AlgoBase
from scipy.stats import beta
from main.utils.integrals import mean_integral
import numpy as np
import pandas as pd


class ORDSTAT(AlgoBase):
    def compute_delta(self):
        # TODO: Use N and confidence to compute delta
        # TODO: Alternative: have a table for N, Confidence and delta
        # NOTE: Current implementation works for only N = 100
        return 0.998

    def compute_cdf(self, N):
        delta = self.compute_delta()

        upper_envelope = lower_envelope = None

        if self.bound == "upper":
            upper_envelope = np.array([beta.ppf(q=(1 + delta) / 2.0, a=i, b=N - i + 1) for i in range(1, N + 1)])
        elif self.bound == "lower":
            lower_envelope = np.array([beta.ppf(q=(1 - delta) / 2.0, a=i, b=N - i + 1) for i in range(1, N + 1)])
        else:
            upper_envelope = np.array([beta.ppf(q=(1 + delta) / 2.0, a=i, b=N - i + 1) for i in range(1, N + 1)])
            lower_envelope = np.array([beta.ppf(q=(1 - delta) / 2.0, a=i, b=N - i + 1) for i in range(1, N + 1)])

        return upper_envelope, lower_envelope

    def compute_mean(self):
        boundsLimits = list()

        for N in xrange(1, self.N + 1):
            upper, lower = self.compute_cdf(N)

            if self.bound == "upper":
                mean = mean_integral(N=N, cdf_values=lower)
                boundsLimits.append([N, mean, 'Upper ORDSTAT', 1])
            elif self.bound == "lower":
                mean = mean_integral(N=N, cdf_values=upper)
                boundsLimits.append([N, mean, 'Lower ORDSTAT', 1])
            else:
                upper_mean = mean_integral(N=N, cdf_values=lower)
                lower_mean = mean_integral(N=N, cdf_values=upper)
                boundsLimits.append([N, upper_mean, 'Upper ORDSTAT', 1])
                boundsLimits.append([N, lower_mean, 'Lower ORDSTAT', 1])

        boundsDF = pd.DataFrame(data=boundsLimits, columns=["N", "Observations", "BoundType", "Unit"])
        return boundsDF
