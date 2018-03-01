import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml


def plot_statistic(df, true_mean=True, **kwargs):
    """
    :param df: pandas DataFrame containing upper/lower bounds for a statistic (mean, variance etc.)
    :param true_mean: Plot True mean if True.
    :return: matplotlib axis
    """
    with open("../resources/config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    err_style = cfg['plot_configs']['err_style']
    interpolate = cfg['plot_configs']['interpolate']
    distribution = cfg['sample_statistics']['distribution']
    T = cfg['sample_statistics']['T']

    N = kwargs['N']
    plt.figure(figsize=(20, 20))

    plt.subplot()

    if true_mean:
        if distribution == 'exponential':
            mean = cfg['exponential']['true_mean']
        elif distribution == 'normal':
            mean = cfg['normal']['true_mean']
        elif distribution == 'uniform':
            mean = cfg['uniform']['true_mean']

        plt.plot(np.arange(1, N+1), [mean]*N, label="True Mean", color='black')

    try:
        assert isinstance(df, pd.DataFrame)
        assert df.columns == ["N", "Observations", "BoundType", "Unit"]
    except Exception as e:
        print e

    sns.tsplot(data=df, time="N", value="Observations", condition="BoundType", unit="Unit", ci=[100], err_style=err_style
            , interpolate=interpolate, n_boot=T)

    plt.title("Bounds on " + kwargs['statistic'])
    plt.show()

