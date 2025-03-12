import numpy as np
import pandas as pd

class BHBAnalyzer():
    """
    A class for performing Brinson-Hood-Beebower (BHB) performance attribution analysis.


    Args
        portfolio_weights : np.ndarray
            A numpy array of portfolio weights across different sectors or stocks.
        benchmark_weights : np.ndarray
            A numpy array of benchmark weights across the same sectors or stocks.
        portfolio_returns : np.ndarray
            A numpy array of portfolio returns for each sector or stock.
        benchmark_returns : np.ndarray
            A numpy array of benchmark returns for each sector or stock.
        allocation_effect : float
            The calculated allocation effect from the analysis.
        selection_effect : float
            The calculated selection effect from the analysis.
        interaction_effect : float
            The calculated interaction effect from the analysis.
        total_effect : float
            The total performance attribution, combining all three effects.

    Methods
        run_analysis()
            Performs the BHB performance attribution analysis and stores the results.
        get_results()
            Returns the results of the performance attribution analysis as a dictionary.
    """

    
    def __init__(self, 
                 benchmark_data=None,
                 experiment_data=None,
                 raw_data=None,
                 n_trading_times = None,
                 ):

        """
        Initializes the Brinson-Hood-Beebower (BHB) analysis class with benchmark and experimental data.
        
        Args:
            benchmark_data : list or np.ndarray
                The benchmark weights for each trading time. 
                Should be structured as a list of arrays or a 2D array where rows represent trading times.
            
            experiment_data : list or np.ndarray
                The experimental (portfolio) weights for each trading time.
                Should be structured as a list of arrays or a 2D array where rows represent trading times.
            
            raw_data : list of lists of np.ndarray
                The historical stock return data for each sector and stock.
                Should be structured as a nested list where raw_data[i][j] corresponds to the returns of stock j in sector i.
            
            n_trading_times : int
                The total number of trading intervals for which the analysis will be conducted.
        
        Attributes:
            num_sectors : int
                The number of sectors in the dataset, derived from the length of raw_data.
            
            num_stocks : int
                The total number of stocks in the dataset, calculated as the number of sectors multiplied by the number of stocks per sector.
            
            num_stocks_per_sector : int
                The number of stocks per sector, assuming an even distribution of stocks across sectors.
            
            periods : int
                The number of trading periods, determined from the benchmark data length.
            
            arit_excess_return : list
                A list storing the allocation, selection, interaction effects, and total excess returns for each trading time.
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
        Performs frequency-based Brinson-Hood-Beebower (BHB) analysis over multiple trading intervals.

        Returns:
        None
            The results of the analysis are stored in the 'arit_excess_return' attribute as a list of lists, 
            where each sublist contains:
            - Allocation Effect (float): The impact of sector weight deviations between benchmark and portfolio.
            - Selection Effect (float): The impact of stock selection skill within each sector.
            - Interaction Effect (float): The combined effect of allocation and selection decisions.
            - Total Excess Return (float): The sum of all effects for the interval.
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
    print("--Frequency BHB analysis performed successfully--")

        