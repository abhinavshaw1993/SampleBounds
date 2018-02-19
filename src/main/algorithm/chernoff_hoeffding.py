from algo_base import AlgoBase
from main.utils import sample_generator
import numpy as np
import seaborn as sns
import pandas as pd

class ChenoffHoeffding(AlgoBase):
    """
    Class for Chernoff-Hoeffding Bounds
    """
    def compute_mean(self):

        sg = sample_generator.SampleGenerator(self.N, self.T)
        samples = sg.normal()



