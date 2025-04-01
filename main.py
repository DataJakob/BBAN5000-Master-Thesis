import numpy as np
import pandas as pd

from src.Data_Retriever import DataRetriever as DatRet

from src.Optimization.Markowitz_PT import MarkowitzPT as MPT

from src.Optimization.Environment import PortfolioEnvironment as PorEnv
from src.Optimization.RLModelCompilation import RL_Model as RLM
from src.Optimization.NeuralNet import CustomNeuralNet as CusNN
from src.Optimization.NeuralNet import CustomSACPolicy as CSACP

from src.Result.Menchero_OGA import MencheroOGA as MOGA
from src.Result.IndPortResults import GenerateResult as GR
from src.Result.OverviewResults import ResultConveyor as RC


"""------------------------------------------------"""
# Define necessary non-fixed variables
trading_n = 400
history_usage = 252
n_sectors = 6
n_stocks_per_sector = 4

# For RL algorithm
history_usage_RL = 50
rolling_reward_window = 50
"""------------------------------------------------"""
# Defining stock pool
ticker_df =  pd.DataFrame()
ticker_df["Petroleum"] = ["EQNR.OL", "AKRBP.OL", "SUBC.OL", "BWO.OL",]
ticker_df["Seafood (food)"] = ["ORK.OL", "MOWI.OL", "SALM.OL", "LSG.OL"]
ticker_df["Materials"] = ["NHY.OL", "YAR.OL", "RECSI.OL", "BRG.OL"]  #del this
ticker_df["Technologies"] = ["TEL.OL", "NOD.OL", "ATEA.OL", "BOUV.OL"]
ticker_df["Financial"] = ["STB.OL", "DNB.OL", "GJF.OL", "AKER.OL"]
ticker_df["Shipping"] = ["WAWI.OL", "SNI.OL", "BELCO.OL", "ODF.OL"]
ticker_df
"""------------------------------------------------"""
# Defining ESG scores for respective securities
esg_scores = np.array([36.6, 35.3, 17.9, 18, 
                18, 21.2, 18.7, 29.2, 
                15.7, 25.6, 25.6, 18.4, # Del this
                19.8, 13.8, 18.1, 19, 
                17.2, 14, 17.2, 19.5, 
                19.7, 21.2, 26.8, 19.3])
# """------------------------------------------------"""
# # Retrieve data from yf API: y-m-d
# data = DatRet(ticker_df, "2013-01-01", "2024-12-31", history_usage_RL=history_usage_RL)
# # In function below, set log=True to check for data availability
# data.retrieve_data()
# # """------------------------------------------------"""
# # Generate benchmark weights thorugh MPT using Sharpe ratio
# benchmark = MPT(history_usage, trading_n)
# # IMPORTANT: In order to see  the effect of the weights, algo exclude last observation from optimization
# benchmark.frequency_optimizing()
# # """------------------------------------------------"""
# objectives = ["Return", "Sharpe", "Sortino", "Sterling", "Return", "Sharpe", "Sortino", "Sterling"]
# esg_compliancy = [True, True, True, True, False, False, False, False]
objectives = ["Sortino"]
esg_compliancy = [True]

for i in range(len(objectives)):
    reinforcement = RLM(esg_scores, 
                        objective=objectives[i],
                        history_usage=history_usage_RL,
                        rolling_reward_window=rolling_reward_window,
                        total_timesteps=1000,
                        esg_compliancy=esg_compliancy[i],
                        )
    reinforcement.train_model()
    reinforcement.test_model()
"""------------------------------------------------"""
paths = ["Return_esg_True", "Sharpe_esg_True",
         "Sortino_esg_True","Sterling_esg_True",
         "Return_esg_False", "Sharpe_esg_False",
         "Sortino_esg_False","Sterling_esg_False",]

analysis_list = []
for i in range(len(paths)):
    att_anal = GR(paths[i],
            n_sectors, n_stocks_per_sector,
            trading_n,
            esg_scores, 
            ticker_df.columns)
    att_anal.friple_frequency_analysis()
    analysis_list.append(att_anal)
"""------------------------------------------------"""
theta = RC(analysis_list, trading_n)
theta.convey_results()
"""------------------------------------------------"""
