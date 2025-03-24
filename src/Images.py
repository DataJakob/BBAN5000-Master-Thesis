import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Menchero_OGA import MencheroOGA as MOGA



class GenerateResult():

    def __init__(self, n_sectors, n_stock_per_sector, n_optimizations, esg_data, sector_names):    
        self.returns = pd.read_csv("../Data/StockReturns.csv")
        self.bench_w = pd.read_csv("../Data/MPT_weights.csv")
        self.return_w = pd.read_csv("../Data/RL_weights_Return.csv")
        self.sharpe_w = pd.read_csv("../Data/RL_weights_Sharpe.csv")
        self.sortino_w = pd.read_csv("../Data/RL_weights_Sortino.csv")
        self.sterling_w = pd.read_csv("../Data/RL_weights_Sterling.csv")
        self.esg_data = esg_data

        self.n_sectors = n_sectors
        self.n_stock = n_stock_per_sector
        self.n_optimizations = n_optimizations
        self.sector_names = sector_names

        self.return_analysis: dict = None
        self.sharpe_analysis: dict = None
        self.sortino_analysis: dict = None
        self.sterling_analysis: dict = None
    
    def store_values(self,i,pa,ps,ar,er,br,esg):
        mydict = {"sector_allocation":pa,
                "sector_selection":ps,
                "active_return": ar,
                "return":er,
                "bench_return":br,
                "esg_score":esg,
                }
        if i==0:
            self.return_analysis = mydict
        elif i==1:
            self.sharpe_analysis = mydict
        elif i==2:
            self.sortino_analysis = mydict
        else:
            self.sterling_analysis = mydict



    def plot_values(self,algo_name, pa, ps, ar, er,br,esg):
        pap = [np.prod(pa[i]+1) for i in range(len(pa))]
        psp = [np.prod(ps[i]+1) for i in range(len(ps))]

        bigfig, ax = plt.subplots(3,2,figsize=(10,10))
        ax[0,0].plot(br, color="grey", label="Benchmark")
        ax[0,0].plot(er, color="blue", label="Experimental")
        ax[0,0].plot(ar, color="green", label= "Geometric active return")
        ax[0,0].set_ylabel("Return")
        ax[0,0].set_xlabel("Trading times")
        ax[0,0].set_title('General Portfolio Performance')
        ax[0,0].legend()

        ax[0,1].plot(er, color="blue", label="Experimental")
        ax[0,1].scatter(x=np.linspace(0,self.n_optimizations-1,self.n_optimizations), y =(br*ar), s=5, color="black")
        # ax[0,1].plot((br*ar)+(0.001*(br*ar)), 
        #             color="grey", linestyle="--", label="Validity control")
        ax[0,1].set_ylabel("Return")
        ax[0,1].set_xlabel("Trading times")
        ax[0,1].set_title('Benchmark * Active return')
        ax[0,1].legend()

        data_arrays = [pap, psp]
        data_labels = ["Allocation", "Selection"]
        ax[1,0].boxplot(data_arrays, tick_labels=data_labels)
        ax[1,0].axhline(y=1, color="black")
        ax[1,0].set_ylabel("Return")
        ax[1,0].set_title('Attribution Variation')

        ax[1,1].plot(esg, color="blue", label="Mean ESG score")
        ax[1,1].set_ylabel("ESG score")
        ax[1,1].set_xlabel("Trading times")
        ax[1, 1].set_title('ESG Score Development')
        ax[1,1].legend()

        ax[2,0].boxplot(pa)
        ax[2,0].axhline(y=0, color="black")
        ax[2,0].set_xticklabels(self.sector_names, rotation=45) 
        ax[2,0].set_title('Allocation Variation by Sector')

        ax[2,1].boxplot(ps)
        ax[2,1].axhline(y=0, color="black")
        ax[2,1].set_xticklabels(self.sector_names, rotation=45) 
        ax[2,1].set_title('Selection Variation by Sector')

        plt.suptitle("Complete Proto Plot for "+algo_name+" Algo", fontsize=12)
        bigfig.tight_layout(pad=2.0)
        bigfig.savefig("../Results/"+algo_name+".PNG", dpi=300, bbox_inches="tight")
        plt.close()




    def friple_frequency_analysis(self):
        objective_df = [self.return_w, self.sharpe_w, self.sortino_w, self.sterling_w]
        objective_storage = ["return_analysis", "sharpe_analysis", "sortino_analysis", "sterling_analysis"]

        bench_w = [self.bench_w.iloc[-self.n_optimizations+time] for time in range(self.n_optimizations)]
        returns = np.array([self.returns.iloc[-self.n_optimizations+time] for time in range(self.n_optimizations)])+1

        for i in range(0,len(objective_df),1):
            exper_w = [objective_df[i].iloc[-self.n_optimizations+time,:] for time in range(self.n_optimizations)]
            analysis = MOGA(None, None,None)
            analysis = MOGA(objective_df[i], n_sectors=self.n_sectors, n_stocks_per_sector=self.n_stock)
            analysis.frequency_analyser()

            exper_returns = np.cumprod([np.dot(exper_w[i],returns[i]) for i in range(len(exper_w))])
            bench_returns = np.cumprod([np.dot(bench_w[i],returns[i]) for i in range(len(bench_w))])

            port_all = analysis.allocation_effects.reshape(-1,self.n_sectors)
            port_all_prod = [np.prod(port_all[i]+1) for i in range(len(port_all))]
            port_sel = analysis.selection_effects.reshape(-1,self.n_sectors)
            port_sel_prod = [np.prod(port_sel[i]+1) for i in range(len(port_sel))]

            active_return = np.cumprod([port_sel_prod[i]*port_all_prod[i] for i in range(self.n_optimizations)])
            average_esg = [exper_w[i]@self.esg_data for i in range(self.n_optimizations)]

            self.store_values(i, 
                              port_all, port_sel, 
                              active_return, 
                              exper_returns, bench_returns, 
                              average_esg)
            
            self.plot_values(objective_storage[i],
                             port_all, port_sel,
                             active_return,
                             exper_returns, bench_returns,
                             average_esg)
        print("----Analysis completed succesfully----")

