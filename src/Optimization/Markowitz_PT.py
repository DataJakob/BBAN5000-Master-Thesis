import numpy as np
import pandas as pd

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize



class MarkowitzPT():
    """
    A class implementing a frequency-based Markowitz Portfolio Theory (MPT) optimization framework.
    
    Attributes
        data : list, nested array with sectors X stocks
        history_usage : int, number of historical data points to use for each optimization.
        n_optimizations : int, number of optimization intervals to perform over the historical data.
        num_stocks : int, number of stocks across all sectors.
        num_sectors : int, number of sectors in the portfolio.
        opt_results : list,  results of the most recent portfolio optimization: [weights, expected return, risk].
        returns : list, expected returns calculated for each optimization interval.
        frequency_weights : list, optimized weights for each time interval

    Methods
        optimize_portfolio(new_data=None)
            Optimizes a portfolio using Mean-Variance Optimization (MVO) based on provided stock data.
        generate_new_positions()
            Generates new stock positions for each time interval using MPT and stores them in `frequency_weights`.
    """

    def __init__(self,  history_usage=None, n_optimizations=None):
        self.data = pd.read_csv("Data/StockReturns.csv")
        self.history_usage: int = history_usage
        self.n_optimizations: int = n_optimizations

        self.num_stocks: int = len(self.data.iloc[0])

        self.opt_results: list = []

        self.returns: list = []
        self.frequency_weights:list = []
        
        

    def optimize_portfolio(self, new_data):
        """
        Optimize a stock portfolio using Mean-Variance Optimization (MVO) and the Sharpe ratio.
        
        Args
            new_data : list, list containing n number of returns  for all stocks in portfolio
        
        Returns
            opt_results : list, list containing the following elements:
            - w : numpy.ndarray, optimized weights for each stock in the portfolio.
            - ret : float, expected return of the optimized portfolio.
            - risk : float, standard deviation of the optimized portfolio.       
        """

        # Generate a list of means
        mean_list = new_data.mean()
        # Create a default covariance matrix   
        cov_matrix = new_data.cov()

        c1 = Bounds(0,1)
        c2 = LinearConstraint(np.ones((self.num_stocks,), dtype=int),1,1)
        weights =  np.ones(self.num_stocks)
        decVar = weights / np.sum(weights)
        
        Z = lambda w: np.sqrt(max(w @ cov_matrix @ w.T, 0))
        res = minimize(Z, decVar, method="trust-constr", constraints=c2, bounds=c1)
        w = res.x
        ret = sum(w*mean_list)
        risk = (w@cov_matrix@w.T)**.5
        opt_results = [w, ret, risk]

        return opt_results
    


    def frequency_optimizing(self):
        """
        Generate new stock positions for each time interval using frequency trading and MPT.

        Attributes
            frequency_weights : list, optimized weights for n times.

        Returns
            str: A success message indicating that the data was successfully generated.
        """

        sliced_data = []
        for time in range(self.n_optimizations,0,-1):
            data_per_time_interval = self.data.iloc[-(self.history_usage+self.n_optimizations+1+time):-(self.n_optimizations+1+time)]
            sliced_data.append(data_per_time_interval)

        counter = 0
        frequency_weights_list = []
        for y in range(0, self.n_optimizations,1): 
            # print(frequency_weights_list)
            if (y % 100) == 0: 
                counter = y     # Rebalancing every second months
                # ind_weights = self.optimize_portfolio(sliced_data[counter])[0]
                # ind_weights = (ind_weights**0.25) /  np.sum(ind_weights**0.25)
                # ind_weights = np.clip(ind_weights, 0.5/18, 3.5/18) / np.sum(np.clip(ind_weights, 0.5/18, 3.5/18))
                
                # Equal weight portfolio
                ind_weights = np.ones(18)/18
            else: 
                ind_weights = frequency_weights_list[-1] * (1+self.data.iloc[-(self.history_usage+self.n_optimizations)+y]) 
                ind_weights = ind_weights / np.sum(ind_weights)
            frequency_weights_list.append(ind_weights)
        self.frequency_weights = frequency_weights_list

        only_weights = pd.DataFrame([self.frequency_weights[i] for i in range(self.n_optimizations)]) # first index = 0 or i
        only_weights.to_csv('Data/MPT_weights.csv', index=False)
        
        print("--Frequency trading using MPT successfully performed--")