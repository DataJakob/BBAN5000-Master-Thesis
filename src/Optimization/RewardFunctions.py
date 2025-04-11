import numpy as np
import pandas


x = 1
upper = 30
lower = 30


def sharpe_ratio(return_window: np.array):
    """
    doc string
    """
    mean = np.mean(return_window)
    stddev = np.std(return_window)

    sharpe = mean / (stddev + 1e-8)
    #sharpe = sharpe * x
    sharpe = np.clip(sharpe, lower, upper) # Redundant, but gives equality to measurements
    if np.isnan(sharpe):
        return 0.0 

    return sharpe  



def return_ratio(return_window: np.array):
    """
    doc string
    """
    #cumu = (np.cumprod(return_window+1)-1)[-1]
    mean = np.mean(return_window)
    #mean = mean * x
    mean = np.clip(mean, lower, upper) # Redundant, but gives equality to measurements

    if np.isnan(mean):
        return 0.0 

    return mean 



def sortino_ratio(return_window: np.array):
    """
    doc string
    """
    mean = np.mean(return_window)
    downside_risk = np.sqrt(np.mean(np.square(np.minimum(return_window, 0))))

    sortino = mean / (downside_risk + 1e-8)
    #sortino = sortino * x
    sortino = np.clip(sortino, lower, upper) # Currently arbitrary

    if np.isnan(sortino):
        return 0.0 

    return sortino 


def sterling_ratio(return_window: np.array):
    """ 
    doc string 
    """
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


def penalise_reward(reward, esg_score):
    """
    doc string
    """
    penalty = 0.3 * (esg_score / 40) 
   
    penalised_reward = reward - penalty     
    
    return penalised_reward