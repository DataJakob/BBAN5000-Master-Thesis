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
    def __init__(self, data=None,  history_usage=None, n_optimizations=None):
        self.data = data
        self.history_usage: int = history_usage
        self.n_optimizations: int = n_optimizations

        self.num_stocks: int = len(data) * len(data[0])
        self.num_sectors: int = len(data)
        self.num_stocks_per_sector = int(self.num_stocks / self.num_sectors)

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
        simple_array = [pd.Series(stock) for sector in new_data for stock in sector]
        dataframe = pd.DataFrame(simple_array)
        transpose_df = dataframe.T

        # Generate a list of means
        mean_list = transpose_df.mean()

        # Create a default covariance matrix    
        cov_matrix = transpose_df.cov()


        c1 = Bounds(0,1)
        c2 = LinearConstraint(np.ones((self.num_stocks,), dtype=int),1,1)
        weights =  np.ones(self.num_stocks)
        decVar = weights / np.sum(weights)
        print("decvar:",decVar)

        Z = lambda w: np.sqrt(w@cov_matrix@w.T)
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
        for sector in self.data:  # For each sector
            sector_sliced = []
            for stock in sector:  # For each stock
                time_sliced = []
                for time in range(self.data[0][0].shape[0]): # For each time interval
                    # Relevant histroical data for stock to be optimized
                    data_per_time_interval = stock[-self.history_usage-self.n_optimizations+time-1:-self.n_optimizations+time-1]
                    time_sliced.append(data_per_time_interval)
                sector_sliced.append(time_sliced)  
            sliced_data.append(sector_sliced)  
            # Sector X Stock X n observation


        frequency_weigths_list = []
        for y in range(0, self.n_optimizations,1):
            
            selective_time_data = [sliced_data[i][j][y] for i in range(self.num_sectors) for j in range(int(self.num_stocks/self.num_sectors))]
            ideal_matrix_format = [selective_time_data[i:i+int(self.num_stocks/self.num_sectors)] for i in range(0, len(selective_time_data), self.num_sectors)]
            print("mat form:", len(ideal_matrix_format))
            print("mat form:", len(ideal_matrix_format[0][0]))

            ind_weights = self.optimize_portfolio(ideal_matrix_format)
            frequency_weigths_list.append(ind_weights)
        self.frequency_weights = frequency_weigths_list
        
        print("--Frequency trading using MPT successfully performed--")