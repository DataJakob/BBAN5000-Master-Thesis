import pandas as pd
import numpy as np

def _calculate_return(returns):
    returns = np.array(returns).flatten() 
    returns = np.cumprod(returns)[-1] - 1
    return np.round(returns,4) * 100

def _calculate_avg_return(returns):
    returns = np.array(returns).flatten() -1
    returns = np.mean(returns) * 10000
    return np.round(returns,3)

def _calculate_volatility(returns):
    returns = np.array(returns).flatten() -1
    volatility = np.std(returns) *10000
    return np.round(volatility,3)

def _calculate_maxdrawdown(returns):
    returns = np.array(returns).flatten() -1
    cumu = (np.cumprod(returns +1))
    peak = np.maximum.accumulate(cumu)
    drawdown = (cumu - peak)/peak
    mdd = np.min(drawdown)
    return np.round(mdd,4)

def _calculate_var(returns):
    returns = np.array(returns).flatten() -1
    returns = np.array(returns)
    returns = returns[returns < 0]
    sorted_returns = np.sort(returns)
    var = np.quantile(sorted_returns, 0.05)
    return np.round(var,4)

def _calculate_cvar(returns):
    returns = np.array(returns).flatten() -1
    returns = returns[returns < 0]
    sorted_returns = np.sort(returns)
    cvar = np.mean(returns[returns < np.quantile(sorted_returns, 0.05)])
    return np.round(cvar,4)


def _calculate_sharpe(returns, risk_free_rate):
    """Calculate annualized Sharpe Ratio"""
    # returns = np.array(returns).flatten() -1
    excess_returns = np.array(returns).flatten() -1 
    mean = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1
    # mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns) * np.sqrt(521)
    if std_excess == 0:
        return 0.0
    return np.round(mean / std_excess ,3)

def _calculate_sortino(returns, risk_free_rate):
    """Calculate annualized Sortino Ratio"""
    mean = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1
    returns = np.array(returns).flatten() -1
    excess_returns = returns - risk_free_rate

    # mean_excess = np.mean(excess_returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_std = np.std(downside_returns) * np.sqrt(521)
    if downside_std == 0:
        return 0.0
    return np.round(mean / downside_std,3)


def _calculate_sterling(returns: np.array):
    """
    Sterling ratio calculator
    """
    
    periodic_return = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1

    all_drawdown = []
    ind_drawdown = []
    for i in range(0, len(returns), 1):
        if returns[i] < 1:
            ind_drawdown.append(returns[i])
            if i == len(returns) - 1:
                all_drawdown.append(ind_drawdown)
        else:
            if len(ind_drawdown) == 0:
                pass
            else:
                all_drawdown.append(ind_drawdown)
                ind_drawdown = []

    prod_drawdowns = np.abs(np.array([np.cumprod(all_drawdown[i])[-1] for i in range(len(all_drawdown))])-1)


    if -int(len(prod_drawdowns) *0.1) == 0:
        idx = len(prod_drawdowns)
    else: 
        idx= -int(len(prod_drawdowns) * 0.1)
    avg_drawdown = np.mean(np.sort(prod_drawdowns)[::][idx:])

    sterling_ratio = periodic_return / avg_drawdown

    return np.round(sterling_ratio,3)

def _calculate_calmar(returns):
    """Calculate Calmar Ratio"""
    mean = (np.cumprod(np.array(returns).flatten()))[-1]**(521/800)-1
    returns = np.array(returns).flatten()-1
    cumu = (np.cumprod(returns +1))
    peak = np.maximum.accumulate(cumu)
    drawdown = (cumu - peak)/peak
    max_drawdown = np.min(drawdown)
    if max_drawdown == 0:
        return np.inf 
    else:
        return np.round(mean / abs(max_drawdown),3)
