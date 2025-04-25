import numpy as np
import pandas

# Upper and lower bounds reward function in clipping mechanisms
lower = -10
upper = 10



def sharpe_ratio(return_window: np.array):
    """
    Calculates a modified Sharpe ratio from a given window of portfolio returns.

    The Sharpe ratio is the average return divided by the standard deviation of returns.
    This implementation adds a scaling factor, clips the output between a specified range,
    and incorporates the most recent daily return change into the final score.

    Args:
        return_window (np.array): A 1D array of past portfolio returns.

    Returns:
        float: A scaled, clipped, and adjusted Sharpe ratio, averaged with the most recent daily return change.
    """
    if len(return_window) <= 2:
        return 0
    
    mean = np.mean(return_window)
    stddev = np.std(return_window)

    sharpe = mean / (stddev + 1e-8)
    sharpe = np.clip(sharpe, lower, upper) 
    if np.isnan(sharpe):
        return 0.0 
    sharpe *= 5

    daily_change =  (return_window[-1] / return_window[-2]) * 100
    return (sharpe + daily_change) / 2
    # return sharpe



def return_ratio(return_window: np.array):
    """
    Computes a custom return-based reward using average and recent return change.

    This metric favors higher average returns and combines it with the most recent return jump.
    The final score is clipped and scaled to improve learning stability.

    Args:
        return_window (np.array): A 1D array of past portfolio returns.

    Returns:
        float: A scaled and averaged score based on mean return and daily return change.
    """

    if len(return_window) < 2:
        return 0
    mean = np.mean(return_window)
    mean = np.clip(mean, lower, upper) 

    if np.isnan(mean):
        return 0.0 
    mean *= 1000

    daily_change =  (return_window[-1] / return_window[-2]) *100
    return (mean + daily_change) / 2
    # return mean



def sortino_ratio(return_window: np.array):
    """
    Calculates a modified Sortino ratio from a given return window.

    The Sortino ratio is a variation of the Sharpe ratio using downside deviation.
    This version includes a daily return change to increase responsiveness and prevent
    reward flattening in periods of low volatility.

    Args:
        return_window (np.array): A 1D array of past portfolio returns.

    Returns:
        float: A scaled and averaged Sortino ratio, adjusted with recent daily change.
    """

    if len(return_window) < 2:
        return 0  
    mean = np.mean(return_window)
    downside_risk = np.sqrt(np.mean(np.square(np.minimum(return_window, 0))))

    sortino = mean / (downside_risk + 1e-8)
    sortino = np.clip(sortino, lower, upper) # Currently arbitrary
    if np.isnan(sortino):
        return 0.0 
    daily_change =  (return_window[-1] / return_window[-2]) *100

    return (sortino + daily_change) / 2



def sterling_ratio(return_window: np.array):
    """ 
    Computes a modified Sterling ratio using drawdowns and average returns.

    The Sterling ratio considers drawdowns from peak cumulative returns to
    assess downside risk. This implementation includes average drawdown and
    appends a daily change for added variability in reward signals.

    Args:
        return_window (np.array): A 1D array of past portfolio returns.

    Returns:
        float: A clipped and averaged Sterling ratio value adjusted with recent daily return change.
    """

    if len(return_window) < 2:
        return 0
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
    sterling = np.clip(sterling, lower, upper)

    if np.isnan(sterling):
        return 0.0 
    daily_change = (return_window[-1] / return_window[-2]) *100

    return (sterling + daily_change) / 2



def penalise_reward(reward: np.array, 
                    esg_score: float):
    """
    Penalizes the reward score based on ESG compliance.

    This function deducts a penalty from the original reward score
    proportionally to the ESG score. A higher ESG score implies greater penalty,
    making the agent favor lower ESG scores when this is activated.

    Args:
        reward (float): The reward score to be penalized.
        esg_score (float): The ESG score of the portfolio at current step.

    Returns:
        float: The penalized reward.
    """

    penalty = 0.3 * (esg_score / 40) 
   
    penalised_reward = reward - penalty     
    
    return penalised_reward