import numpy as np
import pandas as pd

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

class MarkowitzPT():

    def __init__(self, data=None,  history_usage=None, n_optimizations=None):
        """
        Args:
            data: np.array of data
            history_usage: use x historical data to optimize
        """
        self.data = data
        self.history_usage: int = history_usage
        self.n_optimizations: int = n_optimizations

        self.num_stocks: int = len(data) * len(data[0])
        self.num_sectors: int = len(data)
        self.opt_results: list = []

        self.returns: list = []
        self.frequency_weights:list = []
        
        

    def optimize_portfolio(self, new_data = None):
        """
        Optimizes a portfolio consisting of many stonks using MPT and Sharpe ratio
        """
        # Generate a list of means
        mean_list = []
        for sector in range(len(new_data)):
            sector_mean_list = []
            for stock in range(len(new_data[0])):
                mean = np.mean(new_data[sector][stock])
                sector_mean_list.append(mean)
            mean_list.append(sector_mean_list)
        mean_list = [item for row in mean_list for item in row]
        self.returns = mean_list
    
        # Generate a list of standard deviations
        std_list = []
        for sector in range(len(new_data)):
            sector_std_list = []
            for stock in range(len(new_data[0])):
                std = np.std(new_data[sector][stock], axis=0)
                sector_std_list.append(std.iloc[0])
            std_list.append(sector_std_list)
        std_list = [item for row in std_list for item in row]

        # Create a default covariance matrix    
        flat_series_list = [series for row in new_data for series in row]
        cov_matrix = pd.concat(flat_series_list, axis=1).cov()

        c1 = Bounds(0,1)
        c2 = LinearConstraint(np.ones((self.num_stocks,), dtype=int),1,1)
        weights =  np.ones(self.num_stocks)
        decVar = weights / np.sum(weights)

        Z = lambda w: np.sqrt(w@cov_matrix@w.T)
        res = minimize(Z, decVar, method="trust-constr", constraints=c2, bounds=c1)
        w = res.x
        ret = sum(w*mean_list)
        risk = (w@cov_matrix@w.T)**.5

        opt_results = [w, ret, risk]

        return opt_results
    


    def frequency_optimizing(self):
        """
        A function that generates new stock posistions for
        each new time interval, based on a fixed time interval specificed in __init__
        """

        sliced_data = []
        for sector in self.data:  # For each sector
            sector_sliced = []
            for stock in sector:  # For each stock
                time_sliced = []
                for time in range(self.data[0][0].shape[0]): # For each time interval
                    data_per_time_interval = stock.iloc[-self.history_usage-self.n_optimizations+time:-self.n_optimizations+time]
                    time_sliced.append(data_per_time_interval)
                sector_sliced.append(time_sliced)  
            sliced_data.append(sector_sliced)  


        frequency_weigths_list = []
        for y in range(0, self.n_optimizations,1):
            selective_time_data = [sliced_data[i][j][y] for i in range(self.num_sectors) for j in range(int(self.num_stocks/self.num_sectors))]
            ideal_matrix_format = [selective_time_data[i:i+int(self.num_stocks/self.num_sectors)] for i in range(0, len(selective_time_data), self.num_sectors)]
            ind_weights = self.optimize_portfolio(ideal_matrix_format)
            frequency_weigths_list.append(ind_weights)
        self.frequency_weights = frequency_weigths_list
    
        print("--Frequency trading using MPT successfully performed--")