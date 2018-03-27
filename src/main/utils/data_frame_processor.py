"""
This module is responsible for processing the DataFrame for plotting different confidence
intervals and mean of the bounds.
"""
import pandas as pd
import yaml

class ProcessDataframe:

    def __init__(self, df):
        # Initializing the raw dataframe to be processed.
        self.df = df
        self.result_df = pd.DataFrame()

    def process_dataframe(self, df):
        """
        This function processes raw DataFrame returning the DataFrame required for plotting in
        ts plot
        :param df: Data Frame to be processed.
        :return: Processed Data Frame.
        """
        # Reading the Config File.
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        process_mean = cfg['data_processing_details']['process_mean']
        process_var = cfg['data_processing_details']['process_var']
        process_percentiles = cfg['data_processing_details']['process_percentiles']
        process_ts = cfg['data_processing_details']['process_ts']

        if process_mean:
            self.process_mean()
        if process_var:
            self.process_var()
        if process_percentiles:
            self.process_percentiles()
        if process_ts:
            self.process_ts()

        return self.result_df

    def process_percentiles(self):
        """
        This function processes percentiles
        :return: Processed Data Frame.
        """

        # Reading the Config File.
        with open("../resources/config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)

        # Inheriting some configurations for percentile processing.
        percentiles = cfg['data_processing_details']['percentiles']
        T = cfg['sample_statistics']['T']

        # Processing Percentiles.
        sorted_upper = self.df.loc[self.df['BoundType'].str.contains("Upper")].sort_values("Observations")
        sorted_lower = self.df.loc[self.df['BoundType'].str.contains("Lower")].sort_values("Observations", ascending=False)
        sorted_lower.to_csv("/home/abhinav/Desktop/SortedLower.csv")

        for p in percentiles:
            percentile = int(T * p / 100)
            print("Percentile values", percentile)

            # processing Tth Percentile of the Upper Bounds.(There may be multiple bounds here.)
            grouped_upper = sorted_upper.groupby(["N", "BoundType"], as_index=False)
            grouped_lower = sorted_lower.groupby(["N", "BoundType"], as_index=False)
            percentile_data = grouped_lower.nth(percentile)
            percentile_data = percentile_data.append(grouped_upper.nth(percentile), ignore_index=True)
            percentile_data["BoundType"] = percentile_data["BoundType"] + " " + str(p) + "th Percentile"
            percentile_data["Unit"] = 1
            self.result_df = self.result_df.append(percentile_data, ignore_index=True)
            self.result_df.to_csv("/home/abhinav/Desktop/percentile.csv")

    def process_mean(self):
        """
        This function processes mean values.
        """
        # Applying some transformations for mean here.
        mean_observations = self.df.groupby(["BoundType", "N"], as_index=False)["Observations"].mean()
        mean_observations["BoundType"] =  mean_observations["BoundType"] + " Mean"
        mean_observations["Unit"] = 1
        self.result_df = self.result_df.append(mean_observations, ignore_index=True)
        self.result_df.to_csv("/home/abhinav/Desktop/mean.csv")

    def process_ts(self):
        """
        This function processes original ts plot for Seaborn.
        """

        # For the std ts plot we just need to use the original data frame.
        self.result_df = self.result_df.append(self.df, ignore_index=True)

    def process_var(self):
        """
        This function processes variance values.
        """

        # Applying some transformations for Variance here.
        var_observations = self.df.groupby(["BoundType", "N"], as_index=False)["Observations"].var()
        var_observations["BoundType"] =  var_observations["BoundType"] + " Var"
        var_observations["Unit"] = 1
        self.result_df = self.result_df.append(var_observations, ignore_index=True)