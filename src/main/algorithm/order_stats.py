from algo_base import AlgoBase
from scipy.stats import beta
from main.utils.integrals import mean_integral_v2
from main.utils.integrals import variance_integral
import numpy as np
import pandas as pd


class ORDSTAT(AlgoBase):

    def compute_delta(self, N):
        # TODO: Use N and confidence to compute delta
        # TODO: Alternative: have a table for N, Confidence and delta
        # NOTE: Current implementation works for only N = 100
        delta_values = pd.read_csv("../resources/delta_values.csv")
        delta = delta_values[delta_values["N"] == N]["delta"]
        return np.float(delta)

    def compute_cdf(self, N):

        delta = self.compute_delta(N)

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

            samples = self.sample_generator.generate_samples(N, self.T)
            ord_stats = np.sort(samples, 0)
            upper, lower = self.compute_cdf(N)

            for trials in xrange(self.T):
                if self.bound == "upper":
                    mean = mean_integral_v2(cdf_values=lower, ord_stats=ord_stats[:, trials], bound_type="upper")
                    boundsLimits.append([N, mean, 'Upper ORDSTAT', trials + 1])
                elif self.bound == "lower":
                    mean = mean_integral_v2(cdf_values=upper, ord_stats=ord_stats[:, trials], bound_type="lower")
                    boundsLimits.append([N, mean, 'Lower ORDSTAT', trials + 1])
                else:
                    upper_mean = mean_integral_v2(cdf_values=lower, ord_stats=ord_stats[:, trials], bound_type="upper")
                    lower_mean = mean_integral_v2(cdf_values=upper, ord_stats=ord_stats[:, trials], bound_type="lower")

                    boundsLimits.append([N, upper_mean, 'Upper ORDSTAT', trials + 1])
                    boundsLimits.append([N, lower_mean, 'Lower ORDSTAT', trials + 1])

        boundsDF = pd.DataFrame(data=boundsLimits, columns=["N", "Observations", "BoundType", "Unit"])
        return boundsDF

# similar to compute mean
    def compute_variance(self):
        boundsLimits = list()

        for N in xrange(1, self.N + 1):

            samples = self.sample_generator.generate_samples(N, self.T)
            ord_stats = np.sort(samples, 0)
            upper, lower = self.compute_cdf(N)

            for trials in xrange(self.T):
                if self.bound == "upper":
                    mean = variance_integral(cdf_values=lower, ord_stats=ord_stats[:, trials], bound_type="upper")
                    boundsLimits.append([N, mean, 'Upper ORDSTAT', trials + 1])
                elif self.bound == "lower":
                    mean = variance_integral(cdf_values=upper, ord_stats=ord_stats[:, trials], bound_type="lower")
                    boundsLimits.append([N, mean, 'Lower ORDSTAT', trials + 1])
                else:
                    upper_mean = variance_integral(cdf_values=lower, ord_stats=ord_stats[:, trials], bound_type="upper")
                    lower_mean = variance_integral(cdf_values=upper, ord_stats=ord_stats[:, trials], bound_type="lower")

                    boundsLimits.append([N, upper_mean, 'Upper ORDSTAT', trials + 1])
                    boundsLimits.append([N, lower_mean, 'Lower ORDSTAT', trials + 1])
        print ("boundsLimits obtained")
        boundsDF = pd.DataFrame(data=boundsLimits, columns=["N", "Observations", "BoundType", "Unit"])
        return boundsDF