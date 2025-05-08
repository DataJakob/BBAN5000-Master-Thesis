import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr




class ResultConveyor():
    """
    A class for performing performance attribution analysis on experimental 
    portfolio weights compared to benchmark weights. This includes evaluating 
    returns, ESG characteristics, and generating visualization plots for 
    allocation and selection effects.
    
    Parameters
    ----------
    analysis_list : list
        List of analysis objects containing experimental and benchmark results.
    n_optimizations : int
        Number of optimization steps / trading periods.
    """



    def __init__(self, 
                 analysis_list: list, 
                 n_optimizations: int):
        """
        Initialize the ResultConveyor class.
        """
        self.analysis_list = analysis_list
        # All optimizations weights are to be multiplied with returns for time t+1
        self.n_optimizations = n_optimizations -1



    def overview_plot(self):
        """
        Generate a scatter plot comparing selection and allocation effects 
        of various portfolios. Points are colored by ESG use and sized by active return.
        """
        
        sel_port = []
        all_port = []
        act_port = []

        for item in self.analysis_list:
            sel_eff = np.std([np.prod(item.exper_analysis["sector_selection"][i]+1) for i in range(self.n_optimizations)]) - np.mean(item.exper_analysis["sector_selection"]-1) 
            all_eff = np.std([np.prod(item.exper_analysis["sector_allocation"][i]+1) for i in range(self.n_optimizations)]) - np.mean(item.exper_analysis["sector_allocation"]-1)
            act_ret = item.exper_analysis["active_return"][-1]
            sel_port.append(sel_eff)
            all_port.append(all_eff)
            act_port.append(act_ret)

        def minmax(arr):
            arr = np.array(arr)
            relation = 100 / (max(arr)- min(arr)+1e-9)
            arr = arr - min(arr)
            arr *= relation
            return arr
        
        txt = ["Return", "Sharpe", "Sortino", "Sterling", "Calamar"]
        fig, ax = plt.subplots()
        ax.scatter(x=np.array(all_port)+0, y=np.array(sel_port)+0, 
                color='black',
                s=minmax(act_port)*5,
                alpha=0.5,           
                )
        for i in range(5):
            plt.annotate(txt[i],
                (all_port[i]+0.0,sel_port[i]+0.0),
            )
        plt.xlabel(r'($\sigma$'+"-"+r"$\mu$)"+" Allocation")
        plt.ylabel(r'($\sigma$'+"-"+r"$\mu$)"+" Selection")
        plt.title("Portfolio comparison: Attribution", fontsize=16, fontweight='bold', pad=20)
        plt.suptitle("size = active return", y=0.92, fontsize=12, style='italic')
        plt.grid()
        plt.savefig("ResultS/OverviewPlot.PNG")
        plt.close()
    


    def risk_table(self):
        """
        doc string 
        """
        returns = [self.analysis_list[i].exper_analysis["esg_score"] for i in range(5)]
        txt = ["Ret", "Sha", "Sor", "Ste", "Cal"]
        risk_df = pd.DataFrame()
        risk_df["Measurement"] = ["Ret", "AvgRet", "Std", "MDD", "VaR", "CVaR"]

        def _calculate_return(returns):
            returns = np.array(returns).flatten() 
            returns = np.cumprod(returns)[-1] - 1
            return np.round(returns,4) * 100

        def _calculate_avg_return(returns):
            returns = np.array(returns).flatten() -1
            returns = np.mean(returns) * 10000
            return np.round(returns,3)

        def _calculate_volatility(returns):
            returns = np.array(returns).flatten() -1
            volatility = np.std(returns) *10000
            return np.round(volatility,3)

        def _calculate_maxdrawdown(returns):
            returns = np.array(returns).flatten() -1
            cumu = (np.cumprod(returns +1))
            peak = np.maximum.accumulate(cumu)
            drawdown = (cumu - peak)/peak
            mdd = np.min(drawdown)
            return np.round(mdd,4)
        
        def _calculate_var(returns):
            returns = np.array(returns).flatten() -1
            returns = np.array(returns)
            returns = returns[returns < 0]
            sorted_returns = np.sort(returns)
            var = np.quantile(sorted_returns, 0.05)
            return np.round(var,4)
        
        def _calculate_cvar(returns):
            returns = np.array(returns).flatten() -1
            returns = returns[returns < 0]
            sorted_returns = np.sort(returns)
            cvar = np.mean(returns[returns < np.quantile(sorted_returns, 0.05)])
            return np.round(cvar,4)


        for i in range(5):
            risk_df[str(txt[i])] = [_calculate_return(returns[i]),
                                    _calculate_avg_return(returns[i]),
                                    _calculate_volatility(returns[i]),
                                    _calculate_maxdrawdown(returns[i]),
                                    _calculate_var(returns[i]),
                                    _calculate_cvar(returns[i])]
        risk_df.to_csv("Results/risk_table.csv", index=False)



    def financial_table(self):
        """
        Generate a CSV table with financial metrics: P/L, Sharpe, Sortino, and Sterling ratios.
        """

        returns = [self.analysis_list[i].exper_analysis["esg_score"] for i in range(5)]
        txt = ["Ret", "Sha", "Sor", "Ste", "Cal"]
        financial_df = pd.DataFrame()
        financial_df["Measurement"] = ["Sharpe", "Sortino", "Sterling", "Calmar"]


        def _calculate_sharpe(returns, risk_free_rate):
            """Calculate annualized Sharpe Ratio"""
            # returns = np.array(returns).flatten() -1
            excess_returns = np.array(returns).flatten() -1 
            mean = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1
            # mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns) * np.sqrt(521)
            if std_excess == 0:
                return 0.0
            return np.round(mean / std_excess ,3)

        def _calculate_sortino(returns, risk_free_rate):
            """Calculate annualized Sortino Ratio"""
            mean = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1
            returns = np.array(returns).flatten() -1
            excess_returns = returns - risk_free_rate

            # mean_excess = np.mean(excess_returns)
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            downside_std = np.std(downside_returns) * np.sqrt(521)
            if downside_std == 0:
                return 0.0
            return np.round(mean / downside_std,3)


        def _calculate_sterling(returns: np.array):
            """
            Sterling ratio calculator
            """
            
            periodic_return = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1

            all_drawdown = []
            ind_drawdown = []
            for i in range(0, len(returns), 1):
                if returns[i] < 1:
                    ind_drawdown.append(returns[i])
                    if i == len(returns) - 1:
                        all_drawdown.append(ind_drawdown)
                else:
                    if len(ind_drawdown) == 0:
                        pass
                    else:
                        all_drawdown.append(ind_drawdown)
                        ind_drawdown = []

            prod_drawdowns = np.abs(np.array([np.cumprod(all_drawdown[i])[-1] for i in range(len(all_drawdown))])-1)


            if -int(len(prod_drawdowns) *0.1) == 0:
                idx = len(prod_drawdowns)
            else: 
                idx= -int(len(prod_drawdowns) * 0.1)
            avg_drawdown = np.mean(np.sort(prod_drawdowns)[::][idx:])

            sterling_ratio = periodic_return / avg_drawdown

            return np.round(sterling_ratio,3)
        
        def _calculate_calmar(returns):
            """Calculate Calmar Ratio"""
            mean = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1
            returns = np.array(returns).flatten()-1
            cumu = (np.cumprod(returns +1))
            peak = np.maximum.accumulate(cumu)
            drawdown = (cumu - peak)/peak
            max_drawdown = np.min(drawdown)
            if max_drawdown == 0:
                return np.inf 
            else:
                return np.round(mean / abs(max_drawdown),3)


        for i in range(5):
            financial_df[str(txt[i])] = [_calculate_sharpe(returns[i], 0.0),
                                        _calculate_sortino(returns[i], 0.0),
                                        _calculate_sterling(returns[i]),
                                        _calculate_calmar(returns[i])]
        financial_df.to_csv("Results/financial_table.csv", index=False)



    def actvie_return_table(self):
        txt = [ "Ret", "Sha", "Sor", "Ste", "Cal"]
        active_df = pd.DataFrame()
        active_df["Measurment"] = ["Active return", "mu(all)", "sig(all)", "mu(sel)", "sig(sel)"]

        counter= 0
        for item in self.analysis_list:
            ar = np.mean(item.exper_analysis["active_return"])#[-1]
            all_mean = np.median(item.exper_analysis["sector_allocation"])
            all_std = np.std([np.prod(item.exper_analysis["sector_allocation"][i]+1) for i in range(self.n_optimizations)])
            sel_mean = np.median(item.exper_analysis["sector_selection"])
            sel_std = np.std([np.prod(item.exper_analysis["sector_selection"][i]+1) for i in range(self.n_optimizations)])
            active_df[str(txt[counter])] = [np.round((np.array(ar)-1)*100000,2), 
                                            np.round(all_mean*1000000,2), np.round(all_std*1000,2),
                                            np.round(sel_mean*1000000,2), np.round(sel_std*1000,2)]
            counter+=1
        active_df.to_csv("Results/active_df.csv", index=False)


    # def esg_table(self):
    #     """
    #     Generate a CSV table with ESG metrics: average ESG score and 
    #     correlation with returns.
    #     """

    #     txt = ["Ret_ESG","Sha_ESG","Sor_ESG", "Ste_ESG", "Ret", "Sha", "Sor", "Ste"]
    #     esg_df = pd.DataFrame()
    #     esg_df["Measurment"] = ["Avg ESG", "Pearsons R", "p value"]
        
    #     counter = 0
    #     for item in self.analysis_list:
    #         esg_scores = item.exper_analysis["esg_score"]
    #         avg_esg = round(np.mean(esg_scores), 2)
    #         correlation, p_value = pearsonr(esg_scores, item.exper_analysis["return"])
    #         esg_df[txt[counter]] = [avg_esg, round(correlation, 3), round(p_value,3)]
    #         counter += 1

    #     esg_df.to_csv("Results/esg_table.csv", index=False)



    def convey_results(self):
        self.overview_plot()
        self.financial_table()
        self.actvie_return_table()
        self.risk_table()
        # self.esg_table()