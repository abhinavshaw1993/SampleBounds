import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_statistic(df, **kwargs):
    """
    :param df: pandas DataFrame containing upper/lower bounds for a statistic (mean, variance etc.)
    :return: matplotlib axis
    """
    mean = kwargs.get('mean')
    N = kwargs.get('N')
    plt.figure(figsize=(5, 5))
    if mean:
        plt.plot(np.arange(1, N+1), [mean]*N, label="True Mean")

    try:
        assert isinstance(df, pd.DataFrame)
        assert df.columns == ["N", "Observations", "BoundType", "Unit"]
    except Exception as e:
        print e

    plt.subplot()
    sns.tsplot(data=df, time="N", value="Observations", condition="BoundType", unit="Unit", ci=100)
    # sns.factorplot(x='N', y='Observations', hue='BoundType', data=df, size=10)
    plt.show()

