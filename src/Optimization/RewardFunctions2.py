import numpy as np

x = 5 #magnitude multiplier


def sharpe_ratio(return_window: np.array):

    mean = np.mean(return_window)
    stddev = np.std(return_window)

    sharpe = mean / (stddev + 1e-8)

    return sharpe * x  




def return_ratio(return_window: np.array):
    
    cumu = (np.cumprod(return_window+1)-1)[-1]

    return cumu * x



def sortino_ratio(return_window: np.array):

    mean = np.mean(return_window)
    downside_risk = np.sqrt(np.mean(np.square(np.minimum(return_window, 0))))
    downside_risk += 1

    sortino = mean / downside_risk

    return sortino * x


def sterling_ratio(return_window: np.array):

    mean = np.mean(return_window)

    cumu = (np.cumprod(return_window)+1)-1
    peak = cumu.cummax()
    drawdown = (cumu - peak)/peak
    avg_drawdown = np.mean(-drawdown[drawdown < 0])

    sterling = mean / np.abs((avg_drawdown + 1e-8)- 0.1)

    return sterling * x


def penalise_reward(reward_window, esg_score):

    penalty = 0.3 * (esg_score) 
   
    penalised_reward = reward_window - penalty     
    return penalised_reward

