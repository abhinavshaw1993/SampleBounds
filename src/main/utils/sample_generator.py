"""
This module is responsible for generating samples from a fixed set of distributions
All distributions have a fixed positive support [left, right] such that 0<=left<=right
Samples from the following distributions can be generated
 - Normal distribution
 - Beta distribution
 - Exponential distribution
 - Uniform distribution

The module will no start returning the mean as well, since it is hard to control the mean

"""
from scipy.stats import truncnorm
from scipy.stats import truncexpon
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import laplace
from scipy.stats import t
import numpy as np


class SampleGenerator:
    # TODO: Add code/methods for generating distributions

    def __init__(self, left, right, mean, stdev, distribution, random_seed):
        """
        :param N: Number of samples
        :param T: Number of times to perform sampling
        :param mean: mean of the samples, used for truncated normal distribution
        :param stdev: Standard Deviation, used for truncated normal distribution
        """

        self.left = left
        self.right = right
        self.distribution = distribution
        self.random_seed = random_seed
        self.mean = mean
        self.stdev = stdev
        # TODO: Add distribution relevant parameters like mean, std, alpha, beta etc.

    def generate_samples(self, N, T):
        if self.distribution == "normal":
            return self.normal(N, T)
        elif self.distribution == "uniform":
            return self.uniform(N, T)
        elif self.distribution == "exponential":
            return self.exponential(N, T)
        elif self.distribution == "beta":
            return self.beta(N, T)
        elif self.distribution == "mix_gauss_2_sym_uni":
            return self.mix_gauss_2_sym_uni(N, T)
        elif self.distribution == "mix_gauss_2_sym_multi":
            return self.mix_gauss_2_sym_multi(N, T)
        elif self.distribution == "mix_gauss_2_nonsym_uni":
            return self.mix_gauss_2_nonsym_uni(N, T)
        elif self.distribution == "mix_gauss_2_nonsym_multi":
            return self.mix_gauss_2_nonsym_multi(N, T)
        elif self.distribution == "mix_gauss_4_sym_uni":
            return self.mix_gauss_4_sym_uni(N, T)
        elif self.distribution == "mix_gauss_4_sym_multi":
            return self.mix_gauss_4_sym_multi(N, T)
        elif self.distribution == "mix_gauss_4_nonsym_uni":
            return self.mix_gauss_4_nonsym_uni(N, T)
        elif self.distribution == "mix_gauss_4_nonsym_multi":
            return self.mix_gauss_4_nonsym_multi(N, T)
        elif self.distribution == "student_t_3":
            return self.student_t_3(N, T)
        elif self.distribution == "student_t_5":
            return self.student_t_5(N, T)
        elif self.distribution == "double_exponential":
            return self.double_exponential(N, T)
        elif self.distribution == "mix_double_exponential":
            return self.mix_double_exponential(N, T)
        else:
            raise NotImplementedError

    def normal(self, N, T):
        """
        :return: samples from the normal distribution with support [self.left, self.right]
        """

        a, b = (self.left - self.mean) / self.stdev, (self.right - self.mean) / self.stdev
        samples = truncnorm.rvs(a=a, b=b, loc=self.mean, scale=self.stdev, size=(N, T), random_state=self.random_seed)
        return samples

    def uniform(self, N, T):
        """
        :return: samples from the uniform distribution with support [self.left, self.right]
        """

        a = self.left
        b = self.right
        samples = uniform.rvs(loc=self.left, scale=self.right, size=(N, T), random_state=self.random_seed)

        return samples

    def exponential(self, N, T):
        """
        :return: samples from the exponential distribution with support [self.left, self.right]
        Notes - We'll be using exponential with a support 0, 1. To achieve this we'll be
        truncating the exponential distribution after 1. The support of the exponential
        distribution is from 0 to infinity by default. We will be keeping the scale low so that
        not a lot of samples are truncated from the distribution and we have some consistency.
        """

        # truncexpon moves from 0 to b.
        return truncexpon.rvs(b=self.right, loc=self.mean, scale=self.stdev, size=(N, T), random_state=self.random_seed)

    def beta(self, N, T):
        """
        :return: samples from the beta distribution with with the following parameters
        a=0.5, b=1, loc=left support, scale=right support (Already tested with these params, they give us a beta with 0,1 support)
        """
        return beta.rvs(a=0.5, b=1, loc=self.left, scale=self.right, size=(N, T), random_state=self.random_seed)

    def student_t_3(self, N, T):
        return t.rvs(df=3, loc=0.5, scale=0.005, size=(N, T), random_state=self.random_seed)

    def student_t_5(self, N, T):
        return t.rvs(df=5, loc=0.5, scale=0.005, size=(N, T), random_state=self.random_seed)

    def double_exponential(self, N, T):
        mean = 0.5
        stdev = 0.02
        return laplace.rvs(loc=mean, scale=stdev, size=(N, T), random_state=self.random_seed)

    def mix_double_exponential(self, N, T):
        means = [0.4, 0.6]
        stdevs = [0.02, 0.02]
        prop = np.array([.5, .5])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            samples[req_idxs] = laplace.rvs(loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_2_sym_uni(self, N, T):
        """
        :return: Returns samples from a mixture of 2 Gaussian distributions with 0, 1 support
        """

        means = [0.25, 0.75]
        stdevs = [0.5, 0.5]
        prop = np.array([0.5, 0.5])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_2_sym_multi(self, N, T):
        """
        :return: Returns samples from a mixture of 2 Gaussian distributions with 0, 1 support
        """
        means = [0.25, 0.75]
        stdevs = [0.1, 0.1]
        prop = np.array([0.5, 0.5])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_2_nonsym_uni(self, N, T):
        """
        :return: Returns samples from a mixture of 2 Gaussian distributions with 0, 1 support
        """

        means = [0.15, 0.75]
        stdevs = [0.4, 0.4]
        prop = np.array([1., 2.])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_2_nonsym_multi(self, N, T):
        """
        :return: Returns samples from a mixture of 2 Gaussian distributions with 0, 1 support
        """

        means = [0.25, 0.75]
        stdevs = [0.1, 0.1]
        prop = np.array([1., 3.])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_4_sym_multi(self, N, T):
        """
        :return: Returns samples from a mixture of 4 Gaussian distributions with 0, 1 support
        """

        means = [0.2, 0.4, 0.6, 0.8]
        stdevs = [0.05, 0.05, 0.05, 0.05]
        prop = np.array([1., 2., 2., 1.])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_4_sym_uni(self, N, T):
        """
        :return: Returns samples from a mixture of 4 Gaussian distributions with 0, 1 support
        """

        means = [0.15, 0.4, 0.6, 0.85]
        stdevs = [0.2, 0.3, 0.3, 0.2]
        prop = np.array([1., 2., 2., 1.])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_4_nonsym_multi(self, N, T):
        """
        :return: Returns samples from a mixture of 4 Gaussian distributions with 0, 1 support
        """

        means = [0.16666667, 0.6, 0.4, 0.86666667]
        stdevs = [0.05, 0.05, 0.05, 0.05]
        prop = np.array([1., 1., 2., 1.])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def mix_gauss_4_nonsym_uni(self, N, T):
        """
        :return: Returns samples from a mixture of 4 Gaussian distributions with 0, 1 support
        """

        means = [0.1, 0.4, 0.6, 0.75]
        stdevs = [0.13, 0.2, 0.2, 0.11]
        prop = np.array([1., 2., 2., 1.])
        prop = prop / sum(prop)

        samples = np.zeros((N, T))
        np.random.seed(self.random_seed)
        idxs = np.random.choice(np.arange(len(means)), size=(N, T), p=prop)

        for i in range(len(means)):
            req_idxs = idxs == i
            num_samples = np.sum(req_idxs)
            a, b = (self.left - means[i]) / stdevs[i], (self.right - means[i]) / stdevs[i]
            samples[req_idxs] = truncnorm.rvs(a=a, b=b, loc=means[i], scale=stdevs[i], size=num_samples, random_state=self.random_seed)

        return samples

    def true_mean(self):
        def true_mean_normal():
            a, b = (self.left - self.mean) / self.stdev, (self.right - self.mean) / self.stdev
            return truncnorm.mean(a, b, loc=self.mean, scale=self.stdev)

        def true_mean_uniform():
            return (self.right - self.left) / 2.0

        def true_mean_exponential():
            return truncexpon.mean(b=self.right, loc=self.mean, scale=self.stdev)

        def true_mean_beta():
            return beta.mean(a=0.5, b=1, loc=self.left, scale=self.right)

        def true_mean_student_t_3():
            return t.mean(df=3, loc=0.5, scale=0.005)

        def true_mean_student_t_5():
            return t.mean(df=5, loc=0.5, scale=0.005)

        def true_mean_double_exponential():
            return 0.5

        def true_mean_mix_double_exponential():
            return 0.5

        def true_mean_mix_gauss_2_sym_uni():
            return 0.5

        def true_mean_mix_gauss_2_sym_multi():
            return 0.5

        def true_mean_mix_gauss_2_nonsym_uni():
            return 0.55

        def true_mean_mix_gauss_2_nonsym_multi():
            return 0.625

        def true_mean_mix_gauss_4_sym_multi():
            return 0.5

        def true_mean_mix_gauss_4_sym_uni():
            return 0.5

        def true_mean_mix_gauss_4_nonsym_uni():
            return 0.4750

        def true_mean_mix_gauss_4_nonsym_multi():
            return 0.4867

        compute_true_mean = {
            "normal": true_mean_normal,
            "uniform": true_mean_uniform,
            "exponential": true_mean_exponential,
            "beta": true_mean_beta,
            "student_t_3": true_mean_student_t_3,
            "student_t_5": true_mean_student_t_5,
            "double_exponential": true_mean_double_exponential,
            "mix_double_exponential": true_mean_mix_double_exponential,
            "mix_gauss_2_sym_uni": true_mean_mix_gauss_2_sym_uni,
            "mix_gauss_2_sym_multi": true_mean_mix_gauss_2_sym_multi,
            "mix_gauss_2_nonsym_uni": true_mean_mix_gauss_2_nonsym_uni,
            "mix_gauss_2_nonsym_multi": true_mean_mix_gauss_2_nonsym_multi,
            "mix_gauss_4_sym_multi": true_mean_mix_gauss_4_sym_multi,
            "mix_gauss_4_sym_uni": true_mean_mix_gauss_4_sym_uni,
            "mix_gauss_4_nonsym_uni": true_mean_mix_gauss_4_nonsym_uni,
            "mix_gauss_4_nonsym_multi": true_mean_mix_gauss_4_nonsym_multi
        }

        return compute_true_mean[self.distribution]()

    def true_variance(self):
        def true_variance_normal():
            a, b = (self.left - self.mean) / self.stdev, (self.right - self.mean) / self.stdev
            return truncnorm.var(a, b, loc=self.mean, scale=self.stdev)

        def true_variance_uniform():
            return np.square(self.right - self.left) / 12.0

        def true_variance_exponential():
            return truncexpon.var(b=self.right, loc=self.mean, scale=self.stdev)

        def true_variance_beta():
            return beta.var(a=0.5, b=1, loc=self.left, scale=self.right)

        def true_variance_student_t_3():
            return t.var(df=3, loc=0.5, scale=0.005)

        def true_variance_student_t_5():
            return t.var(df=5, loc=0.5, scale=0.005)

        def true_variance_double_exponential():
            return 0.5

        def true_variance_mix_double_exponential():
            return 0.5

        def true_variance_mix_gauss_2_sym_uni():
            return 0.3125

        def true_variance_mix_gauss_2_sym_multi():
            return 0.0725

        def true_variance_mix_gauss_2_nonsym_uni():
            return 0.24

        def true_variance_mix_gauss_2_nonsym_multi():
            return 0.05687

        def true_variance_mix_gauss_4_sym_multi():
            return 0.5

        def true_variance_mix_gauss_4_sym_uni():
            return 0.5

        def true_variance_mix_gauss_4_nonsym_uni():
            return 0.4750

        def true_variance_mix_gauss_4_nonsym_multi():
            return 0.4867

        compute_true_variance = {
            "normal": true_variance_normal,
            "uniform": true_variance_uniform,
            "exponential": true_variance_exponential,
            "beta": true_variance_beta,
            "student_t_3": true_variance_student_t_3,
            "student_t_5": true_variance_student_t_5,
            "double_exponential": true_variance_double_exponential,
            "mix_double_exponential": true_variance_mix_double_exponential,
            "mix_gauss_2_sym_uni": true_variance_mix_gauss_2_sym_uni,
            "mix_gauss_2_sym_multi": true_variance_mix_gauss_2_sym_multi,
            "mix_gauss_2_nonsym_uni": true_variance_mix_gauss_2_nonsym_uni,
            "mix_gauss_2_nonsym_multi": true_variance_mix_gauss_2_nonsym_multi,
            "mix_gauss_4_sym_multi": true_variance_mix_gauss_4_sym_multi,
            "mix_gauss_4_sym_uni": true_variance_mix_gauss_4_sym_uni,
            "mix_gauss_4_nonsym_uni": true_variance_mix_gauss_4_nonsym_uni,
            "mix_gauss_4_nonsym_multi": true_variance_mix_gauss_4_nonsym_multi
        }

        return compute_true_variance[self.distribution]()
