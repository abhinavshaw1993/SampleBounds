import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
import itertools


def plot_statistic(df, N, T, true_mean=None, true_variance=None, **kwargs):
    """
    :param df: pandas DataFrame containing upper/lower bounds for a statistic (mean, variance etc.)
    :param true_mean: Plot True mean if True.
    :return: matplotlib axis
    """
    with open("../resources/config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    err_style = cfg['plot_configs']['err_style']
    interpolate = cfg['plot_configs']['interpolate']
    color_type = cfg['plot_configs']['color_type']
    underlying_distribution = cfg['sample_statistics']['distribution']
    plt.figure(figsize=(20, 20))
    plt.subplot()

    if true_mean:
        plt.plot(np.arange(1, N+1), [true_mean]*N, label="True Mean", color='black', linewidth=0.5)

    if true_variance:
        plt.plot(np.arange(1, N+1), [true_variance]*N, label="True Variance", color='black', linewidth=0.5)

    try:
        assert isinstance(df, pd.DataFrame)
        assert df.columns == ["N", "Observations", "BoundType", "Unit"]
    except Exception as e:
        print e

    # Plotting ts plot.
    if color_type == "same":
        # Setting color pallet for different lines
        unique_conditions = list(df.BoundType.unique())
        colors = get_colors_for_plot(unique_conditions)
        ax = sns.tsplot(data=df, time="N", value="Observations", condition="BoundType", unit="Unit", ci=[100], color=colors ,
                        err_style=err_style)
    else:
        ax = sns.tsplot(data=df, time="N", value="Observations", condition="BoundType", unit="Unit", ci=[100], err_style=err_style)
    plt.title("Bounds on " + kwargs['statistic'] + " with " + str(T) + " Trials \n with underlying distribution as " + underlying_distribution)

    # # set_line_width(ax, plt)
    # min_width, max_width = 1, 4
    # lines = ax.lines
    # width = max_width
    #
    # for i in xrange(1, len(unique_conditions), 2):
    #     plt.setp(lines[i], linewidth=width)
    #     plt.setp(lines[i+1], linewidth=width)
    #     width = ((width - min_width) / 2.0) + min_width
    #
    # # # Setting line transparency.
    # # for i in xrange(3, len(unique_conditions), 2):
    # #     lines[i].set_linestyle("--")
    # #     lines[i+1].set_linestyle("--")
    path="C:/Users/Shruti Jadon/Desktop/Plots/Mean/"
    #plt.savefig(path+cfg['sample_statistics']['distribution']+".png")
    plt.show()


def get_colors_for_plot(unique_condition):
    """
    :param unique_condition: List of unique cnoditions in the dataframe for plot.
    :param type: The way you want to set colors.
    :return: Color Dictionary for ts plot.
    """
    pallet = ["green", "red", "windows blue", "amber", "dusty purple"]
    color_pallet = itertools.cycle(sns.xkcd_palette(pallet))
    color_dict = {}
    unique_condition.sort()

    for i in range(len(unique_condition)/2):
        color = next(color_pallet)
        color_dict[unique_condition[i]], color_dict[unique_condition[len(unique_condition)/2+i]] = color, color

    return color_dict


