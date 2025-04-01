import numpy as np



def sharpe_ratio(return_array: np.array, min_obs=3):
    """
    doc string
    """
    n = len(return_array)
    
    # Return 0 if not enough observations
    if n < min_obs:
        return 0.0
    
    # Calculate excess returns
    excess_returns = return_array 
    
    # Calculate mean and stddev
    mean = np.mean(excess_returns)
    stddev = np.std(excess_returns, ddof=1)  # Sample standard deviation
    
    # Avoid division by zero or extreme values
    if stddev < 1e-5:
        return 0.0
    
    # Annualize the sharpe ratio (bidaily returns * 365/2 periods per year)
    sharpe = mean / stddev 
    
    # Cap extreme values to reasonable range [-3, 3]
    sharpe = np.clip(sharpe, -3.0, 3.0)
    
    return sharpe



def sortino_ratio(return_array: np.array, periods_per_year: int = 504) -> float:
    """
    Calculate annualized Sortino ratio with robust numerical handling
    
    Args:
        return_array: Array of arithmetic returns
        periods_per_year: Number of periods in a year (default 504 for 2x daily)
        
    Returns:
        float: Sortino ratio (0.0 for invalid cases)
    """
    # Input validation
    if not isinstance(return_array, np.ndarray):
        return 0.0
        
    return_array = return_array[~np.isnan(return_array)]  # Remove NaNs
    
    if len(return_array) < 2:
        return 0.0
    
    # Calculate mean return with stability checks
    mean_return = np.mean(return_array)
    if np.isnan(mean_return) or np.isinf(mean_return):
        return 0.0
    
    # Handle downside returns
    downside_returns = return_array[return_array < 0]
    
    # Case 1: No downside returns
    if len(downside_returns) == 0:
        return 1000.0 if mean_return > 0 else 0.0  # Large finite value instead of inf
    
    # Case 2: All downside returns are zero (edge case)
    if np.all(downside_returns == 0):
        return 0.0
    
    # Calculate downside volatility with multiple safeguards
    try:
        downside_std = np.std(downside_returns, ddof=1)
        
        # Additional stability checks
        if np.isnan(downside_std) or np.isinf(downside_std):
            return 0.0
            
        downside_std = max(downside_std, 1e-8)  # Prevent division by zero
        
        # Annualize metrics
        annualized_mean = mean_return * periods_per_year
        annualized_downside = downside_std * np.sqrt(periods_per_year)
        
        ratio = annualized_mean / annualized_downside
        
        # Final sanity check
        if np.isnan(ratio) or np.isinf(ratio):
            return 0.0
            
        return float(ratio)
        
    except Exception as e:
        print(f"Sortino ratio calculation error: {str(e)}")
        return 0.0




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
    if len(return_array) < 2:
        return 0.0
    
    periods_per_year = 504
    
    # Annualized return
    annualized_return = np.mean(return_array) * periods_per_year
    
    # Calculate drawdowns
    wealth_index = np.cumprod(1 + return_array)
    previous_peaks = np.maximum.accumulate(wealth_index)
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Get largest 3 drawdowns and average them
    if len(drawdowns) > 0:
        largest_drawdowns = np.sort(drawdowns)[:min(3, len(drawdowns))]
        avg_drawdown = np.mean(largest_drawdowns)
    else:
        avg_drawdown = 0.0
    
    if avg_drawdown >= 0:  # No drawdown occurred
        return np.inf if annualized_return > 0 else 0.0
    
    return annualized_return / abs(avg_drawdown)



def return_ratio(return_array):
    """
    doc string
    """
    if return_array[-1] >= 0:
        reward = return_array[-1]*100
    else:
        reward = return_array[-1] * 100
    return reward


def penalise_reward(reward, esg_score):
    """
    doc string 
    """
    penalty = 0.3 * ((reward/100) * ((esg_score/100)*2.5))   
    penalised_reward = reward - penalty     

    return penalised_reward