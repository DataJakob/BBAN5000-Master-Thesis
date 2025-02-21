import numpy as np
import pandas as pd

class BHBAnalyzer():
    
    def __init__(self, 
                 benchmark_data=None,
                 experiment_data=None,
                 raw_data=None
                 ):
        """
        Args:
            benchmark_data: NxM list from MPT
            experiment_data: NxM list from RL    
        """
        self.benchmark_data =  benchmark_data
        self.experiment_data = experiment_data
        self.raw_data = raw_data
        self.num_sectors: int = len(raw_data)
        self.num_stocks: int = len(raw_data) * len(raw_data[0])
        self.periods: int = len(self.benchmark_data) - 1
        
        self.geo_excess_return: list = []

    def frequency_analyze(self):
        """
        An explanation...
        """
        daily_returns = []
        for time in range(1,self.periods,1):
            ind_day_ret = np.array([self.raw_data[i][j].iloc[-time].iloc[0] for i in  range(self.num_sectors) for j in range(int(self.num_stocks/self.benchmark_data))]) -1
            daily_returns.append(ind_day_ret[::])
        daily_returns = daily_returns[::-1]

        daily_ben_weights = self.benchmark_data[:-1]
        daily_ben_weights = np.array([self.benchmark_data[i][0] for i in range(len(self.benchmark_data))])
        daily_exp_weights = np.repeat(1/9,9)

        rl_perf = [daily_ben_weights[time]@daily_returns[time] for time in range(self.periods)]
        mpt_perf = [daily_exp_weights@daily_returns[time] for time in range(self.periods)]
        excess_ret = [rl_perf[time]-mpt_perf[time] for time in range(self.periods)]
        self.geo_excess_return = excess_ret