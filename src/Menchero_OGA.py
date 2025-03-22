import pandas as pd
import numpy as np



class MencheroOGA():
    """
    A class for performing frequency analysis using the Optimized Geometric Attribution (OGA) model for 
    portfolio performance measurement, made by Menchero (2005).

    Attributes:
        benchmark_w (list): A nested list containing benchmark weights over time for each stock in each sector.
        experimental_w (list): A nested list containing experimental/portfolio weights over time for each stock in each sector.
        returns (list): A nested list of stock returns over time for each stock in each sector.
        n_optimizations (int): The number of time periods for which optimization is performed.
        n_sectors (int): The number of sectors in the portfolio.
        n_stocks (int): The number of stocks per sector.
        allocation_effects (list): A list storing the computed allocation effects over all time periods.
        selection_effects (list): A list storing the computed selection effects over all time periods.
    
    Methods:
        __init__(returns, benchmark_w, experimental_w):
            Initializes the MencheroOGA object with given returns, benchmark weights, and experimental weights.
        
        analyzer_at_time_t(ret, we, wb):
            Computes optimized allocation and selection effects at a given time step.
        
        frequency_analyser():
            Conducts frequency analysis over the entire optimization period and calculates the 
            allocation and selection effects for each time step.
    """

    def __init__(self, n_sectors, n_stocks_per_sector):
        """
        Initializes the MencheroOGA class with portfolio returns, benchmark weights, and experimental weights.

        Args:
            returns (list): A nested list of returns for each stock in each sector over multiple time periods.
            benchmark_w (list): A nested list of benchmark weights corresponding to stocks over multiple time periods.
            experimental_w (list): A nested list of experimental/portfolio weights corresponding to stocks over multiple time periods.
        """
        self.benchmark_w =  pd.read_csv("../Data/MPT_weights.csv")
        self.experimental_w = pd.read_csv("../Data/RL_weights.csv")
        self.returns = pd.read_csv("../Data/StockReturns.csv")

        self.n_optimizations: int = self.benchmark_w.shape[0]
        
        self.n_sectors = n_sectors
        self.n_stocks = n_stocks_per_sector

        self.allocation_effects: list = None
        self.selection_effects: list = None


    def analyzer_at_time_t(self, ret:list,  we:list, wb:list):
        """
        Performs optimized geometric attribution analysis for a single time period.

        Args:
            ret (list): A list of returns for each stock at time t.
            we (list): A list of portfolio weights for each stock at time t.
            wb (list): A list of benchmark weights for each stock at time t.

        Returns:
            list: A list containing two elements:
                  - sel_opt (numpy.ndarray): Optimized selection effects for all sectors at time t.
                  - all_opt (numpy.ndarray): Optimized allocation effects for all sectors at time t.
        """
        # Returns and weights on sector level for  benchmark
        w_b = np.array([sum(wb[int(i*self.n_stocks):int((i+1)*self.n_stocks)]) for i in range(self.n_sectors)])
        r_b = np.array([wb[int(i*self.n_stocks):int((i+1)*self.n_stocks)] @ ret[int(i*self.n_stocks):int((i+1)*self.n_stocks)] for i in range(self.n_sectors)])/w_b

        # Returns and weights on sector level for portfolio
        w_e = np.array([sum(we[int(i*self.n_stocks):int((i+1)*self.n_stocks)]) for i in range(self.n_sectors)])
        r_e = np.array([we[int(i*self.n_stocks):int((i+1)*self.n_stocks)] @ ret[int(i*self.n_stocks):int((i+1)*self.n_stocks)] for i in range(self.n_sectors)])/w_e

        # Total portfolio and benchmark return
        Re = np.dot(w_e, r_e)
        Rb = np.dot(w_b, r_b)

        # Naked allocation and selection effects
        sel_nkd = ((1+w_e*r_e) / (1+ w_e*r_b)) -1
        all_nkd = ((1 + (w_e-w_b)*r_b) /  (1+(w_e-w_b)*Rb)) -1

        # Q value
        q_top =  np.log(1+Re) - np.log(1+Rb) - sum(np.log((1+sel_nkd)*(1+all_nkd)))
        q_bottom = sum(np.log(1+sel_nkd)**2) + sum(np.log(1+all_nkd)**2)
        q_tot =  q_top/q_bottom
        
        # Perturbation terms
        gam_sel  = q_tot * np.log(1+sel_nkd)**2
        gam_all = q_tot * np.log(1+all_nkd)**2

        # Optimized selection and allocation effects
        sel_opt = (1+sel_nkd) * np.e**(gam_sel) - 1
        all_opt = (1+all_nkd) * np.e**(gam_all) - 1

        return [sel_opt, all_opt]        



    def frequency_analyser(self):
        """
        Performs frequency analysis over multiple time periods to calculate selection and allocation effects.

        Returns:
            str: A success message indicating that the data was successfully generated.
        
        Side Effects:
            - Updates the 'allocation_effects' and 'selection_effects' attributes of the class.
        """

        relevant_return_list = [self.returns.iloc[-(self.n_optimizations)+time] for time in range(self.n_optimizations)]
        relevant_exper_list = [self.experimental_w.iloc[-self.n_optimizations+time,1:] for time in range(self.n_optimizations)]

        allocation_list = []
        selection_list = []
        for time in range(self.n_optimizations):
            effects = self.analyzer_at_time_t(relevant_return_list[time], 
                                              np.array(relevant_exper_list[time]),
                                              np.array(self.benchmark_w.iloc[time]))

            selection_list.append(effects[0])
            allocation_list.append(effects[1])
        
        self.allocation_effects = np.concatenate(allocation_list)
        self.selection_effects = np.concatenate(selection_list)
        print("--Frequency analysis performed succesfully--")
