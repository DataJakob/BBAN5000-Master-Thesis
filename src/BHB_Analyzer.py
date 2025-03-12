import numpy as np
import pandas as pd

class BHBAnalyzer():
    
    def __init__(self, 
                 benchmark_data=None,
                 experiment_data=None,
                 raw_data=None,
                 n_trading_times = None,
                 ):
        """
        Args:
            benchmark_data: NxM list from MPT
            experiment_data: NxM list from RL    
        """
        self.benchmark_data =  benchmark_data
        self.experiment_data = experiment_data
        self.raw_data = raw_data

        self.n_trading_times = n_trading_times
        self.num_sectors: int = len(raw_data)
        self.num_stocks: int = len(raw_data) * len(raw_data[0])
        self.num_stocks_per_sector = int(self.num_stocks / self.num_sectors)
        self.periods: int = len(self.benchmark_data) - 1
        
        self.arit_excess_return: list = []

    def frequency_analyze(self):
        """
        An explanation...
        """
        daily_returns = []
        for time in range(1, self.n_trading_times, 1):
            # i=stocks, j=sectors
            ind_day_ret = np.array([self.raw_data[i][j][-time] for i in  range(self.num_sectors) for j in range(self.num_stocks_per_sector)]) -1

            daily_returns.append(ind_day_ret[::])
            daily_returns = daily_returns[::-1]

            daily_ben_weights = self.benchmark_data[:-1]
            daily_ben_weights = np.array([np.array(self.benchmark_data[i][0]) for i in range(len(self.benchmark_data))])[:-1]
            
        # Change this!!!
        daily_exp_weights = [np.repeat(1/9,9) for _ in range(9)]
       
        
        time_list = []
        for trading_time in range(0, self.n_trading_times-1, 1):
            allocation_effect = 0
            selection_effect = 0
            interaction_effect = 0
            
            # Divide portfolio into 3 sectors (0:3, 3:6, 6:9)
            for i in range(self.num_sectors):
                wb = sum(daily_ben_weights[trading_time][i*self.num_stocks_per_sector:(i+1)*self.num_stocks_per_sector])
                we = sum(daily_exp_weights[trading_time][i*self.num_stocks_per_sector:(i+1)*self.num_stocks_per_sector])
                rb = sum(np.array(daily_ben_weights[trading_time][i*self.num_stocks_per_sector:(i+1)*self.num_stocks_per_sector]) * 
                         np.array(daily_returns[trading_time][i*self.num_stocks_per_sector:(i+1)*self.num_stocks_per_sector]))/wb
                re = sum(np.array(daily_exp_weights[trading_time][i*self.num_stocks_per_sector:(i+1)*self.num_stocks_per_sector]) *
                          np.array(daily_returns[trading_time][i*self.num_stocks_per_sector:(i+1)*self.num_stocks_per_sector]))/we


                
                allocation_effect += rb * (we - wb)  # Allocation Effect
                selection_effect += wb * (re - rb)  # Selection Effect
                interaction_effect += (we - wb) * (re - rb)  # Interaction Effect
            
            total_excess_return = allocation_effect + selection_effect + interaction_effect

            time_list.append([allocation_effect, selection_effect, interaction_effect, total_excess_return])
        
        self.arit_excess_return = time_list

        