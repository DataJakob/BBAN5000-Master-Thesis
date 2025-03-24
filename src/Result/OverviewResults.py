import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class ResultConveyor():

    def __init__(self, analysis_list, n_optimizations):
        self.analysis_list = analysis_list
        self.n_optimizations = n_optimizations



    def overview_plot(self):
        sel_port = []
        all_port = []
        act_port = []

        for item in self.analysis_list:
            sel_eff = np.std([np.prod(item.exper_analysis["sector_allocation"][i]+1) for i in range(self.n_optimizations)]) - np.mean(item.exper_analysis["sector_allocation"]-1) 
            all_eff = np.std([np.prod(item.exper_analysis["sector_allocation"][i]+1) for i in range(self.n_optimizations)]) - np.mean(item.exper_analysis["sector_allocation"]-1)
            act_ret = item.exper_analysis["active_return"][-1]
            sel_port.append(sel_eff)
            all_port.append(all_eff)
            act_port.append(act_ret)

        def minmax(arr):
            arr = np.array(arr)
            relation = 100 / (max(arr)- min(arr))
            arr = arr - min(arr)
            arr *= relation
            return arr
        
        txt = ["Return","Sharpe","Sortino", "Sterling", "Return", "Sharpe", "Sortino", "Sterling"]
        fig, ax = plt.subplots()
        ax.scatter(x=np.array(all_port)+0, y=np.array(sel_port)+0, 
                color=["g" if  i < 4 else "r" for i in range(8)],
                s=minmax(act_port)*5,
                alpha=0.5,           
                )
        for i in range(8):
            plt.annotate(txt[i],
                (all_port[i]+0.00005,sel_port[i]+0.0000),
            )
        plt.xlabel(r'($\sigma$'+"-"+r"$\mu$)"+" Allocation")
        plt.ylabel(r'($\sigma$'+"-"+r"$\mu$)"+" Selection")
        plt.title("Portfolio comparison", fontsize=16, fontweight='bold', pad=20)
        plt.suptitle("size = active return, green = esg, red = no esg", y=0.92, fontsize=12, style='italic')
        plt.grid()
        plt.savefig("ResultS/OverviewPlot.PNG")
        plt.close()
    


    def financial_table(self):
        returns = [self.analysis_list[0].exper_analysis["bench_return"]] + [self.analysis_list[i].exper_analysis["active_return"] for i in range(8)]
        txt = ["Benchmark","Ret_ESG","Sha_ESG","Sor_ESG", "Ste_ESG", "Ret", "Sha", "Sor", "Ste"]
        metrics = ["P/L", "Sharpe", "Sortino", "Sterling"]
        financial_df = pd.DataFrame()

        def _calculate_PL(returns):
            """Calculate P/L"""
            return returns[-1] -1 

        def _calculate_sharpe(returns, risk_free_rate):
            """Calculate annualized Sharpe Ratio"""
            returns = returns -1
            excess_returns = returns - risk_free_rate
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns)
            if std_excess == 0:
                return 0.0
            return mean_excess / std_excess 

        def _calculate_sortino(returns, risk_free_rate):
            """Calculate annualized Sortino Ratio"""
            returns = returns -1
            excess_returns = returns - risk_free_rate
            mean_excess = np.mean(excess_returns)
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            downside_std = np.std(downside_returns)
            if downside_std == 0:
                return 0.0
            return mean_excess / downside_std 

        def _calculate_sterling(cumulative_returns):
            """Calculate Sterling Ratio"""
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (peak - cumulative_returns) / (peak + 1e-10) 
            if len(drawdowns) == 0:
                return 0.0
            avg_drawdown = np.mean(drawdowns)
            total_return = cumulative_returns[-1] - 1 
            if avg_drawdown == 0:
                return 0.0
            return total_return / avg_drawdown
        
        for i in range(9):
            financial_df[str(txt[i])] = [_calculate_PL(returns[i]),
                                        _calculate_sharpe(returns[i], 0.0),
                                        _calculate_sortino(returns[i], 0.0),
                                        _calculate_sterling(returns[i])]
        financial_df.to_csv("Results/financial_table.csv")


    def actvie_return_table(self):
        txt = ["Ret_ESG","Sha_ESG","Sor_ESG", "Ste_ESG", "Ret", "Sha", "Sor", "Ste"]
        active_df = pd.DataFrame()

        counter= 0
        for item in self.analysis_list:
            ar = item.exper_analysis["active_return"][-1]
            all_mean = np.mean(item.exper_analysis["sector_allocation"])
            all_std = np.std([np.prod(item.exper_analysis["sector_allocation"][i]+1) for i in range(self.n_optimizations)])
            sel_mean = np.mean(item.exper_analysis["sector_selection"])
            sel_std = np.std([np.prod(item.exper_analysis["sector_selection"][i]+1) for i in range(self.n_optimizations)])
            active_df[str(txt[counter])] = [ar, 
                                            all_mean, all_std,
                                            sel_mean, sel_std]
            counter+=1
        active_df.to_csv("Results/active_df.csv")



    def convey_results(self):
        self.overview_plot()
        self.financial_table()
        self.actvie_return_table()