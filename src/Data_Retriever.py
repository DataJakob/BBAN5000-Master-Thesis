import pandas as pd
import numpy as np

import yfinance as yf



class DataRetriever:
    """
    A class to retrieve, process, and store historical financial data for a list of stock tickers.
    
    This class uses the Yahoo Finance API via the yfinance package to download historical stock
    data, calculate daily returns, volume (z-scored), rolling returns, and rolling volatility. 
    The processed data is stored in separate dataframes and saved to CSV files for later use.

    Attributes:
        start_date (str): Start date of the time window (format: 'YYYY-MM-DD').
        end_date (str): End date of the time window (format: 'YYYY-MM-DD').
        ticker_list (np.ndarray): Flattened array of ticker symbols.
        master_daterange (pd.Series): Business day date range, repeated to simulate intra-day frequency.
        return_df (pd.DataFrame): DataFrame storing percent change (returns) for each ticker.
        volume_df (pd.DataFrame): DataFrame storing standardized volume (z-score) for each ticker.
        rolling_return_df (pd.DataFrame): Rolling 40-day mean return for each ticker.
        rolling_volatility_df (pd.DataFrame): Rolling 40-day standard deviation of returns for each ticker.
    """
    


    def __init__(self, start_date, end_date, ticker_df):
        """
        Initialize the DataRetriever with start/end dates and a DataFrame of tickers.

        Args:
            start_date (str): The starting date for the data retrieval.
            end_date (str): The ending date for the data retrieval.
            ticker_df (pd.DataFrame): DataFrame containing ticker symbols.
        """

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = np.array(ticker_df).T.flatten()

        self.master_daterange = pd.Series(np.repeat(pd.date_range(start=self.start_date, end=self.end_date, freq="B"), 2), name="Date")
        self.return_df = pd.DataFrame(index=self.master_daterange)
        self.volume_df = pd.DataFrame(index=self.master_daterange)
        self.rolling_return_df = pd.DataFrame(index=self.master_daterange)
        self.rolling_volatility_df = pd.DataFrame(index=self.master_daterange)



    def z_score(self, arr: np.array):
        """
        Compute the z-score of a numeric array.

        Args:
            arr (np.array): Input array of numeric values.

        Returns:
            np.array: Z-scored array with mean 0 and standard deviation 1.
        """

        z_score_arr = (arr - np.mean(arr)) / np.std(arr)

        return z_score_arr



    def retrieve_data(self):
        """
        Download and process stock data for each ticker.

        For each ticker:
        - Downloads historical Open, Close, and Volume data.
        - Constructs a time-aligned DataFrame with a custom intra-day format.
        - Computes daily percent returns, z-scored volume, rolling mean returns, and rolling volatility.
        - Saves the processed data into CSV files:
            - 'Data/Input.csv' contains the merged feature set.
            - 'Data/StockReturns.csv' contains only the daily return series.

        Raises:
            Exception if data download fails or format is incompatible.
        """
        
        for i in range(0, len(self.ticker_list), 1):
            stock_data = yf.download(self.ticker_list[i], start=self.start_date, end=self.end_date)

            date_df = pd.DataFrame(index=self.master_daterange)
            open = stock_data["Open"]
            individual_df = pd.merge(date_df, open, on="Date", how="left")
            inter_df = pd.merge(date_df, stock_data["Close"], on="Date", how="left")
            individual_df[self.ticker_list[i]].iloc[1::2] = inter_df[self.ticker_list[i]].iloc[::2]

            return_series = individual_df.pct_change()
            return_series.replace(0.0, np.nan, inplace=True)  # Replace 0.0 with np.nan
            return_series = return_series.interpolate().ffill().bfill() 

            vol_series = pd.merge(date_df, stock_data["Volume"], on="Date", how="left") / 2
            vol_series.replace(0.0, np.nan, inplace=True)  # Replace 0.0 with np.nan
            vol_series = vol_series.interpolate().ffill().bfill()

            self.return_df[self.ticker_list[i]] = return_series[self.ticker_list[i]]
            self.volume_df[self.ticker_list[i]] = self.z_score(vol_series[self.ticker_list[i]])
            self.rolling_return_df[self.ticker_list[i]] = return_series[self.ticker_list[i]].rolling(40).mean().bfill()
            self.rolling_volatility_df[self.ticker_list[i]] = return_series[self.ticker_list[i]].rolling(40).std().bfill()
        
        master_df = pd.concat([self.return_df, self.volume_df, self.rolling_return_df, self.rolling_volatility_df], axis = 1)
        master_df.to_csv("Data/Input.csv", index=False)
        self.return_df.to_csv("Data/StockReturns.csv", index=False)

        print("--- Data retrieved successfully ---")