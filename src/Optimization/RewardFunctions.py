import numpy as np



def sharpe_ratio(return_array: np.array):
    """
    doc string
    """

    ratio =  np.mean(return_array) / (np.std(return_array) + 1e-8)

    return ratio 



def sortino_ratio(return_array: np.array):
    """
    doc string
    """

    annualized_mean_return = np.mean(return_array) 
    
    downside_risk = np.sqrt(np.mean(np.square(np.minimum(return_array, 0))))
    downside_risk += 1

    ratio = annualized_mean_return / downside_risk

    return ratio*1000



def calculate_drawdown(return_array):

    if len(return_array) == 0 or np.all(return_array == 0):
        return 0.005

    with np.errstate(divide='ignore', invalid='ignore'):
        cum_returns = np.cumprod(return_array + 1) - 1
        diff_returns = np.diff(cum_returns)

    drawdown_list = []
    individual_dd = 0

    for i in range(len(diff_returns)):
        if diff_returns[i] < 0:
            individual_dd += diff_returns[i]
        else:
            if individual_dd != 0:
                drawdown_list.append(individual_dd)
            individual_dd = 0

        if i == len(diff_returns)-1 and individual_dd != 0:
            drawdown_list.append(individual_dd)

    sorted_drawdowns = bravo = np.sort(drawdown_list)[::-1]

    if len(sorted_drawdowns) > 10:
        sorted_drawdowns = sorted_drawdowns[:-int(0.1*sorted_drawdowns)]
    elif len(bravo) >= 2:
        sorted_drawdowns = sorted_drawdowns[:-1]
    else: 
        None
    
    if not drawdown_list:
        return 0.005
    
    adjusted_mean_drawdown = np.mean(np.abs(sorted_drawdowns)) + 0.005

    return adjusted_mean_drawdown



def sterling_ratio(return_array):
    """
    doc string
    """
    if len(return_array) == 0 or np.all(return_array == 0):
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        period_return = (np.cumprod(return_array + 1) - 1)[-1]

    drawdown = calculate_drawdown(return_array)

    ratio = period_return / drawdown

    return ratio



def return_ratio(return_array):
    """
    doc string
    """
    reward = (np.cumprod(return_array+1)-1)[-1]
 

    return reward*100



def penalise_reward(reward, esg_score):
    """
    doc string 
    """
    penalty = 0.3 * (esg_score) 
   
    penalised_reward = reward - penalty     
    return penalised_reward