import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_statistic(df):
    """
    :param df: pandas DataFrame containing upper/lower bounds for a statistic (mean, variance etc.)
    :return: matplotlib axis
    """

    try:
        assert isinstance(df, pd.DataFrame)
        assert df.columns == ["N", "Observations", "BoundType", "Unit"]
    except Exception as e:
        # print "error({0}): {1}".format(e.errno, e.strerror)
        print e

    plt.subplot()
    sns.tsplot(data=df, time="N", value="Observations", condition="BoundType", unit="Unit", ci=100)
    plt.show()
