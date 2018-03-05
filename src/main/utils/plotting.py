import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml


def plot_statistic(df, N, T, true_mean=None, **kwargs):
    """
    :param df: pandas DataFrame containing upper/lower bounds for a statistic (mean, variance etc.)
    :param true_mean: Plot True mean if True.
    :return: matplotlib axis
    """
    with open("../resources/config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    err_style = cfg['plot_configs']['err_style']
    interpolate = cfg['plot_configs']['interpolate']
    plt.figure(figsize=(20, 20))
    plt.subplot()

    if true_mean:
        plt.plot(np.arange(1, N+1), [true_mean]*N, label="True Mean", color='black', linewidth=5.0)

    try:
        assert isinstance(df, pd.DataFrame)
        assert df.columns == ["N", "Observations", "BoundType", "Unit"]
    except Exception as e:
        print e

    sns.tsplot(data=df, time="N", value="Observations", condition="BoundType", unit="Unit", ci=[100], err_style=err_style
            , interpolate=interpolate, n_boot=T, color="black")

    plt.title("Bounds on " + kwargs['statistic'])
    plt.show()

