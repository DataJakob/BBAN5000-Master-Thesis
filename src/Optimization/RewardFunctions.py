import numpy as np



def sharpe_ratio(return_array: np.array):
    """
    doc string
    """
    rolling_reward_window = len(return_array)

    annualized_mean_return = np.mean(return_array) * (252*2/rolling_reward_window)
    annualized_std_return = (np.std(return_array) + 1e-8) * np.sqrt(252*2/rolling_reward_window)

    ratio = annualized_mean_return / (annualized_std_return if annualized_std_return != 0 else 1)

    return ratio



def sortino_ratio(return_array: np.array):
    """
    doc string
    """
    rolling_reward_window = len(return_array)
    annualized_mean_return = np.mean(return_array) * (252*2/rolling_reward_window)
    
    downside_risk = np.sqrt(np.mean(np.square(np.minimum(return_array, 0))))
    annualized_downside_risk = downside_risk * np.sqrt(252*2 / rolling_reward_window)

    ratio = annualized_mean_return / (annualized_downside_risk if annualized_downside_risk != 0 else 1)

    return ratio



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
    # reward = return_array[-1]

    return reward



def penalise_reward(reward, esg_score):
    """
    doc string 
    """
    penalty = 0.3 * ((reward/100) * (esg_score*2.5))   

    penalised_reward = reward - penalty     

    return penalised_reward