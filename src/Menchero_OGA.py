import pandas as pd
import numpy as np



class MencheroOGA():
    """
    one-liner

    attributes

    methods
    """

    def __init__(self, returns, benchmark_w, experimental_w):
        """s
        one-liner
        
        Args
        """
        self.benchmark_w =  benchmark_w
        self.experimental_w = experimental_w

        self.n_optimizations: int = len(self.benchmark_w)
        self.returns = returns
        
        self.n_sectors = len(self.returns)
        self.n_stocks = len(self.benchmark_w[0][0]) / self.n_sectors   # Per sector

        self.allocation_effects: list = None
        self.selection_effects: list = None


    def analyzer_at_time_t(self, ret:list,  we:list, wb:list):
        """
        one-liner

        Args

        Returns
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

    # def analyzer_time_t(self, ret:list, we:list, wb:list):
    #     """
    #     one-liner

    #     Args

    #     Returns
    #     """

    #     # Returns and weights on sector level for  benchmark
    #     r_b = np.array([wb[int(i*self.n_stocks):int((i+1)*self.n_stocks)] @ ret[int(i*self.n_stocks):int((i+1)*self.n_stocks)] for i in range(self.n_sectors)])
    #     w_b = np.array([sum(wb[int(i*self.n_stocks):int((i+1)*self.n_stocks)]) for i in range(self.n_sectors)])
        
    #     # Returns and weights on sector level for portfolio
    #     r_e = np.array([we[int(i*self.n_stocks):int((i+1)*self.n_stocks)] @ ret[int(i*self.n_stocks):int((i+1)*self.n_stocks)] for i in range(self.n_sectors)])
    #     w_e = np.array([sum(we[int(i*self.n_stocks):int((i+1)*self.n_stocks)]) for i in range(self.n_sectors)])

    #     # Total portfolio and benchmark return
    #     Re = np.dot(w_e,r_e)
    #     Rb = np.dot(w_b,r_b)

    #     # Naked allocation and selection effects
    #     sel_nkd = ((1+w_e*r_e) / (1+ w_e*r_b)) -1
    #     all_nkd = ((1 + (w_e-w_b)*r_b) /  (1+(w_e-w_b)*Rb)) -1

    #     # Q value, in three equations for simplicity
    #     q_top =  np.log(1+Re) - np.log(1+Rb) - sum(np.log((1+sel_nkd)*(1+all_nkd)))
    #     q_bottom = sum(np.log2(1+sel_nkd)) + sum(np.log2(1+all_nkd))
    #     q_tot =  q_top/q_bottom
        
    #     # Optimized selection and allocation effects, on sector level.
    #     sel_opt = (1+sel_nkd) * np.e**(q_tot*np.log2(1+sel_nkd))
    #     all_opt = (1+all_nkd) * np.e**(q_tot*np.log2(1+all_nkd))

    #     # Returns allocation and slection in a list on sector level
    #     return [all_opt, sel_opt]



    def frequency_analyser(self):
        """
        One liner
        
        Args
        
        Returns
        """
        return_array =  [pd.Series(stock) for sector in self.returns for stock in sector]
        return_df = pd.DataFrame(return_array)
        return_tdf = return_df.T

        # relevant_return_list = [return_tdf.iloc[-+i] for i in range(self.n_optimizations)]
        relevant_return_list = [return_tdf.iloc[-i] for i in range(self.n_optimizations, 0, -1)]
        # relevant_benW_list = self.benchmark_w


        # MOCK DATA SO FAR!!!
        total_n_stocks = self.n_stocks * self.n_sectors
        relevant_expW_list = [[np.repeat(1/total_n_stocks, total_n_stocks)] for i in range(self.n_optimizations)]


        allocation_list = []
        selection_list = []
        for time in range(self.n_optimizations):
            # print(relevant_expW_list[time][0])
            effects = self.analyzer_at_time_t(relevant_return_list[time], 
                                              relevant_expW_list[time][0],
                                              self.benchmark_w[time][0])
            selection_list.append(effects[0])
            allocation_list.append(effects[1])
        
        self.allocation_effects = np.concatenate(allocation_list)
        self.selection_effects = np.concatenate(selection_list)
        print("--Frequency analysis performed succesfully--")