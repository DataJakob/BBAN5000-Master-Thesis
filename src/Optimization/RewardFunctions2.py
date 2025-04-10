import numpy as np
import pandas as pd

x = 1 #magnitude multiplier (tagged out as it fucked the output structure)
upper = 150
lower = -100  # Arbitrary bounds set to limit sortino and sterling from extreem values


def sharpe_ratio(return_window: np.array):

    mean = np.mean(return_window)
    stddev = np.std(return_window)

    sharpe = mean / (stddev + 1e-8)
    #sharpe = sharpe * x
    sharpe = np.clip(sharpe, lower, upper) # Redundant, but gives equality to measurements
    if np.isnan(sharpe):
        return 0.0 

    return sharpe  



def return_ratio(return_window: np.array):
    
    #cumu = (np.cumprod(return_window+1)-1)[-1]
    mean = np.mean(return_window)
    #mean = mean * x
    mean = np.clip(mean, lower, upper) # Redundant, but gives equality to measurements

    if np.isnan(mean):
        return 0.0 

    return mean 



def sortino_ratio(return_window: np.array):

    mean = np.mean(return_window)
    downside_risk = np.sqrt(np.mean(np.square(np.minimum(return_window, 0))))

    sortino = mean / (downside_risk + 1e-8)
    #sortino = sortino * x
    sortino = np.clip(sortino, lower, upper) # Currently arbitrary

    if np.isnan(sortino):
        return 0.0 

    return sortino 


def sterling_ratio(return_window: np.array):

    mean = np.mean(return_window)

    cumu = (np.cumprod(return_window +1))
    peak = np.maximum.accumulate(cumu)
    drawdown = (cumu - peak)/peak
    #avg_drawdown = np.mean(-drawdown[drawdown < 0]) 
    negative_drawdowns = drawdown[drawdown < 0]
    if negative_drawdowns.size == 0:
        avg_drawdown = 0.0  # Returns maximum sterling, given returns
    else:
        avg_drawdown = np.mean(-negative_drawdowns)

    sterling = mean / (avg_drawdown + 1e-8)
    #sterling = sterling * x
    sterling = np.clip(sterling, lower, upper)

    if np.isnan(sterling):
        return 0.0 

    return sterling 


def penalise_reward(reward_window, esg_score):

    penalty = 0.3 * (esg_score) 
   
    penalised_reward = reward_window - penalty     
    return penalised_reward