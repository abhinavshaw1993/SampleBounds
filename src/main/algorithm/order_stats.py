from algo_base import AlgoBase
from scipy.stats import beta
from main.utils.integrals import mean_integral_v2
import numpy as np
import pandas as pd


class ORDSTAT(AlgoBase):

    def compute_delta(self, N):
        # TODO: Use N and confidence to compute delta
        # TODO: Alternative: have a table for N, Confidence and delta
        # NOTE: Current implementation works for only N = 100
        delta_values = pd.read_csv("../resources/delta_values.csv")
        delta = delta_values[delta_values["N"] == N]["delta"]
        # print "delta for " + str(N) + " is: ", np.float(delta)
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

        true_mean = self.sample_generator.true_mean()
        print "true_mean: ", true_mean

        for N in xrange(1, self.N + 1):
            upper_counter = 0.0
            lower_counter = 0.0
            samples = self.sample_generator.generate_samples(N, self.T)
            ord_stats = np.sort(samples, 0)
            upper, lower = self.compute_cdf(N)

            for trials in xrange(self.T):
                if self.bound == "upper":
                    mean = mean_integral_v2(cdf_values=lower, ord_stats=ord_stats[:, trials])
                    boundsLimits.append([N, mean, 'Upper ORDSTAT', trials + 1])
                elif self.bound == "lower":
                    mean = mean_integral_v2(cdf_values=upper, ord_stats=ord_stats[:, trials])
                    boundsLimits.append([N, mean, 'Lower ORDSTAT', trials + 1])
                else:
                    upper_mean = mean_integral_v2(cdf_values=lower, ord_stats=ord_stats[:, trials])
                    lower_mean = mean_integral_v2(cdf_values=upper, ord_stats=ord_stats[:, trials])

                    if upper_mean < true_mean:
                        upper_counter += 1.0

                    if lower_mean > true_mean:
                        lower_counter += 1.0

                    boundsLimits.append([N, upper_mean, 'Upper ORDSTAT', trials + 1])
                    boundsLimits.append([N, lower_mean, 'Lower ORDSTAT', trials + 1])

            print "Lower mean violations % for N = " + str(N) + " is " + str((lower_counter/self.T)*100.0)
            print "Upper mean violations % for N = " + str(N) + " is " + str((upper_counter / self.T) * 100.0)

        boundsDF = pd.DataFrame(data=boundsLimits, columns=["N", "Observations", "BoundType", "Unit"])
        return boundsDF
