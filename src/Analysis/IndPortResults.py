import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.Analysis.Menchero_OGA import MencheroOGA as MOGA



class GenerateResult():
    """
    A class to analyze and visualize the performance of a portfolio optimization algorithm 
    compared to a benchmark. It evaluates returns, attribution effects, and ESG scores, and 
    generates visualizations of performance over time.

    Attributes:
        returns (DataFrame): DataFrame containing stock return data.
        bench_w (DataFrame): Benchmark portfolio weights.
        path (str): Identifier used for locating experiment result files.
        exper_w (DataFrame): Experimental portfolio weights.
        esg_data (array): ESG scores corresponding to each stock.
        n_sectors (int): Number of sectors in the portfolio.
        n_stock (int): Number of stocks per sector.
        n_optimizations (int): Number of trading/optimization periods.
        sector_names (list): List of sector names.
        exper_analysis (dict): Dictionary to store analysis results.
    """



    def __init__(self,
                path: str, 
                n_sectors: int, 
                n_stock_per_sector: int, 
                n_optimizations: int, 
                esg_data: np.array, 
                sector_names: list):    
        """
        Initialize the GenerateResult class.

        Args:
            path (str): Identifier for experiment files.
            n_sectors (int): Number of sectors.
            n_stock_per_sector (int): Number of stocks per sector.
            n_optimizations (int): Number of optimization steps.
            esg_data (array-like): Array of ESG scores for each stock.
            sector_names (list): Names of sectors.
        """
        self.returns = pd.read_csv("Data/StockReturns.csv")
        self.bench_w = pd.read_csv("Data/MPT_weights.csv")
        self.path = path
        self.exper_w = pd.read_csv("Data/TestPredictions/RL_weights_"+self.path+".csv")
        self.esg_data = esg_data

        self.n_sectors = n_sectors
        self.n_stock = n_stock_per_sector
        
        # All optimizations weights are to be multiplied with returns for time t+1
        self.n_optimizations = n_optimizations 
        self.sector_names = sector_names
        
        self.exper_analysis: dict = None


    
    def store_values(self,
                     pa: list,
                     ps: list,
                     ar: np.array,
                     er: np.array,
                     br: np.array,
                     esg: list):
        """
        Store calculated portfolio performance and attribution values.

        Args:
            pa (array-like): Allocation effects per sector.
            ps (array-like): Selection effects per sector.
            ar (array-like): Active return over time.
            er (array-like): Experimental portfolio return.
            br (array-like): Benchmark portfolio return.
            esg (array-like): Average ESG scores over time.
        """

        mydict = {"sector_allocation":pa,
                "sector_selection":ps,
                "active_return": ar,
                "return":er,
                "bench_return":br,
                "esg_score":esg,
                }
        self.exper_analysis = mydict



    def plot_values(self,
                    algo_name: str, 
                    pa: list, 
                    ps: list,
                    ar: np.array, 
                    er: np.array,
                    br: np.array,
                    esg: list):
        """
        Generate and save plots that summarize portfolio performance.

        Args:
            algo_name (str): Name of the algorithm used (used for plot title and filename).
            pa (array-like): Allocation effects per sector.
            ps (array-like): Selection effects per sector.
            ar (array-like): Active return over time.
            er (array-like): Experimental portfolio return.
            br (array-like): Benchmark portfolio return.
            esg (array-like): Average ESG scores over time.
        """

        bigfig, ax = plt.subplots(3,2,figsize=(10,10))
        ax[0,0].plot(br, color="grey", label="Benchmark")
        ax[0,0].plot(er, color="blue", label="Experimental")
        ax[0,0].plot(ar, color="green", label= "Geometric active return")
        ax[0,0].scatter(x=np.linspace(0,self.n_optimizations-1, self.n_optimizations), y=(br*ar), 
                s=5, color="black", label="Validity Control")
        ax[0,0].axhline(y=1, color="red")
        ax[0,0].set_ylabel("Return")
        ax[0,0].set_xlabel("Trading times")
        ax[0,0].set_title('General Portfolio Performance')
        ax[0,0].legend()

        pap = [np.prod(pa[i]+1) for i in range(len(pa))]
        psp = [np.prod(ps[i]+1) for i in range(len(ps))]
        data_arrays = [np.array(pap)-1, np.array(psp)-1]
        data_labels = ["Allocation", "Selection"]
        ax[0,1].boxplot(data_arrays, tick_labels=data_labels)
        ax[0,1].axhline(y=0, color="black")
        ax[0,1].set_ylabel("Return")
        ax[0,1].set_title('Attribution Effect Variation')

        ax[1,0].scatter(x=esg, y=pd.Series(ar).pct_change(),
                        s= 12, alpha=.3,
                        color="black")
        ax[1,0].set_xlabel("Average ESG score")
        ax[1,0].set_ylabel("Portfolio Return")
        ax[1,0].set_title("Correlation: ESG x Return")

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
        bigfig.savefig("Results/"+algo_name+".PNG", dpi=300, bbox_inches="tight")
        plt.close()



    def friple_frequency_analysis(self):
        """
        Conduct full performance analysis:
        - Uses Menchero OGA to compute attribution effects.
        - Calculates cumulative returns and active return.
        - Computes ESG-weighted scores.
        - Stores results and generates visual plots.

        The method also prints a success message after analysis and visualization.
        """
        
        # objective_df = self.exper_w
        # objective_storage = self.path
        # bench_w = [self.bench_w.iloc[-self.n_optimizations+time] for time in range(self.n_optimizations)]
        returns = self.returns.iloc[-self.n_optimizations:].reset_index(drop=True)
        exper_w = self.exper_w.iloc[-self.n_optimizations-1:-1].reset_index(drop=True)
        bench_w = self.bench_w
        analysis = MOGA(exper_w, n_sectors=self.n_sectors, n_stocks_per_sector=self.n_stock)
        analysis.frequency_analyser()

        exper_returns = np.cumprod([np.dot(exper_w.iloc[i], returns.iloc[i])+1 for i in range(self.n_optimizations)])
        bench_returns = np.cumprod([np.dot(bench_w.iloc[i], returns.iloc[i])+1 for i in range(self.n_optimizations)])

        port_all = analysis.allocation_effects.reshape(-1,self.n_sectors)
        port_all_prod = [np.prod(port_all[i]+1) for i in range(len(port_all))]
        port_sel = analysis.selection_effects.reshape(-1,self.n_sectors)
        port_sel_prod = [np.prod(port_sel[i]+1) for i in range(len(port_sel))]

        active_return = np.cumprod([port_sel_prod[i]*port_all_prod[i] for i in range(self.n_optimizations)])
        average_esg = [np.abs(exper_w.iloc[i])@self.esg_data for i in range(self.n_optimizations)]

        self.store_values(self.exper_w, 
                            port_all, port_sel, 
                            active_return, 
                            exper_returns, bench_returns, 
                            average_esg)
        
        self.plot_values(self.path,
                            port_all, port_sel,
                            active_return,
                            exper_returns, bench_returns,
                            average_esg)

        print("----Analysis completed succesfully----")