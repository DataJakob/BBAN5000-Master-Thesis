# Import libraries and custom modules
import numpy as np
import pandas as pd

from src.Data_Retriever import DataRetriever as DatRet

from src.Optimization.Markowitz_PT import MarkowitzPT as MPT

from src.Optimization.Environment import PortfolioEnvironment as PorEnv
from src.Optimization.RLModelCompilation import RL_Model as RLM

from src.Analysis.Menchero_OGA import MencheroOGA as MOGA
from src.Analysis.IndPortResults import GenerateResult as GR
from src.Analysis.OverviewResults import ResultConveyor as RC

import time
"""------------------------------------------------"""
# Start timer
start_time = time.time()
"""------------------------------------------------"""
# Define variables for securitiy pool
n_sectors = 6
n_stocks_per_sector = 3

# Define variables for benchmark
trading_n = 800
history_usage = int(521)

# For RL algorithm
history_usage_RL = 50
rolling_reward_window = 50
"""------------------------------------------------"""
# Defining security pool
ticker_df =  pd.DataFrame({
    "Petroleum": ["EQNR.OL", "SUBC.OL", "BWO.OL"],
    "Food": ["ORK.OL", "MOWI.OL", "LSG.OL"],
    "Materials": ["NHY.OL", "YAR.OL", "RECSI.OL"],
    "Technologies": ["TEL.OL", "NOD.OL", "ATEA.OL"],
    "Financial": ["STB.OL", "DNB.OL", "AKER.OL"],
    "Shipping":["SNI.OL", "BELCO.OL", "ODF.OL"]
})
"""------------------------------------------------"""
# Defining ESG scores for respective securities
esg_scores = np.array([
    36.6, 17.9, 18, 
    18, 23.2, 29.2, 
    15.7, 25.4, 25.6, 
    19.8, 13.8, 18.1, 
    17.3, 14, 12.3, 
    21.2, 26.8, 24.9
])
"""------------------------------------------------"""
# # Retrieve data from yf API: y-m-d
# data = DatRet("2006-07-01", "2024-12-31", ticker_df)
# # In function below, set log=True to check for data availability
# data.retrieve_data()
"""------------------------------------------------"""
# Generate benchmark weights through MPT using Sharpe ratio
benchmark = MPT(history_usage, trading_n)
# IMPORTANT: In order to see  the effect of the weights, algo exclude last observation from optimization
benchmark.frequency_optimizing()
"""------------------------------------------------"""
# Generate production quality (True) portfolio weights
# objectives = ["Return", "Return", 
#               "Sharpe", "Sharpe", 
#               "Sortino", "Sortino", 
#               "Sterling", "Sterling"]

# esg_compliancy = [True, False,
#                   True, False,
#                   True, False,
#                   True, False,]


# esg_compliancy = [False,False,False,False,False]
# objectives = ["Return", "Sharpe", "Sortino", "Sterling", "Calmar"]


# for i in range(len(objectives)):
#     reinforcement = RLM(esg_scores, 
#                         objective=objectives[i],
#                         history_usage=history_usage_RL,
#                         rolling_reward_window=rolling_reward_window,
#                         total_timesteps=200_000,
#                         esg_compliancy=esg_compliancy[i], 
#                         gen_validation_weights=False,
#                         production=True
#                         )
#     reinforcement.train_model()
#     reinforcement.predict()
# """------------------------------------------------"""
# Generate result for individual portfolio and use MOGA
# paths = ["Return_esg_True", "Sharpe_esg_True",
#          "Sortino_esg_True","Sterling_esg_True",
#          "Return_esg_False", "Sharpe_esg_False",
#          "Sortino_esg_False","Sterling_esg_False",
#          "Calmar_esg_False"]
paths = ["Return_esg_False", "Sharpe_esg_False",
         "Sortino_esg_False","Sterling_esg_False",
         "Calmar_esg_False"]

analysis_list = []
for i in range(len(paths)):
    att_anal = GR(
        paths[i],
        n_sectors, 
        n_stocks_per_sector,
        trading_n,
        esg_scores, 
        ticker_df.columns
    )
    att_anal.friple_frequency_analysis()
    analysis_list.append(att_anal)
"""------------------------------------------------"""
# Generalt overview result for comparison
overview_result = RC(analysis_list, trading_n)
overview_result.convey_results()
"""------------------------------------------------"""
# End timer
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")