import numpy as np



def sharpe_ratio(return_array: np.array,
                 min_obs: int=3
                 ):
    """
    doc string
    """
    n = len(return_array)
    if n < min_obs:
        return 0.0
        
    # Calculate mean and stddev
    mean = np.mean(return_array)
    stddev = np.std(return_array) 
    
    # Avoid division by zero or extreme values
    if stddev < 1e-5:
        return mean/1
    
    sharpe =(mean / stddev) 
    # if sharpe < 0:
    #     return -1.0
    sharpe_clipped = np.clip(sharpe, -3.0, 3.0) * 5

    return sharpe_clipped



def sortino_ratio(return_array: np.array, 
                  min_obs: int = 3
                  ):
    """
    doc string
    """
    n = len(return_array)
    if n < min_obs:
        return 0.0
    
    mean = np.mean(return_array) 
    
    downside_risk = np.sqrt(np.mean(np.square(np.minimum(return_array, 0))))
    downside_risk += 1

    sortino = mean / (downside_risk if downside_risk >= 1e-5 else 1)
    sortino_clipped = np.clip(sortino,  -3,3) * 60

    return sortino_clipped




def calculate_drawdown(return_array):
    """
    doc string
    """
    if len(return_array) == 0:
        return 0.0
    
    wealth_index = np.cumprod(1 + return_array)
    
    previous_peaks = np.maximum.accumulate(wealth_index)
    
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    return np.min(drawdowns)  # Maximum drawdown (most negative)


def sterling_ratio(return_array):
    """
    doc string
    """
    # def max_drawdown_penalty(returns):
    cumulative = np.cumsum(return_array)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / (peak + 1e-5)
    max_dd = np.max(drawdown)

    reward =  sharpe_ratio(return_array) -0.5 * (-max_dd) * 5
    return reward
    # if len(return_array) < 2:
    #     return 0.0
    
    # periods_per_year = 504
    
    # # Annualized return
    # annualized_return = np.mean(return_array) * periods_per_year
    
    # # Calculate drawdowns
    # wealth_index = np.cumprod(1 + return_array)
    # previous_peaks = np.maximum.accumulate(wealth_index)
    # drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # # Get largest 3 drawdowns and average them
    # if len(drawdowns) > 0:
    #     largest_drawdowns = np.sort(drawdowns)[:min(3, len(drawdowns))]
    #     avg_drawdown = np.mean(largest_drawdowns)
    # else:
    #     avg_drawdown = 0.0
    
    # if avg_drawdown >= 0:  # No drawdown occurred
    #     return np.inf if annualized_return > 0 else 0.0
    
    # return annualized_return / abs(avg_drawdown)



def return_ratio(return_array):
    """
    doc string
    """
    # if return_array[-1] >= 0:
    #     reward = return_array[-1]*100
    # else:
    #     reward = return_array[-1] * 100
    reward = (np.cumprod(return_array+1)-1)[-1] * 5
    
    return reward


def penalise_reward(reward, esg_score):
    """
    doc string 
    """
    penalty = 0.3 * ((reward/100) * ((esg_score/100)*2.5))   
    penalised_reward = reward - penalty     

    return penalised_reward