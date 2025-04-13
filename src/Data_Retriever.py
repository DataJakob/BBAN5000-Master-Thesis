import pandas as pd
import numpy as np

import yfinance as yf



class DataRetriever:
    """
    doc string 
    """
    
    
    
    def __init__(self, start_date, end_date, ticker_df):
        """
        doc string
        """
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = np.array(ticker_df).flatten()

        self.master_daterange = pd.Series(np.repeat(pd.date_range(start=self.start_date, end=self.end_date, freq="B"), 2), name="Date")
        self.return_df = pd.DataFrame(index=self.master_daterange)
        self.volume_df = pd.DataFrame(index=self.master_daterange)
        self.rolling_return_df = pd.DataFrame(index=self.master_daterange)
        self.rolling_volatility_df = pd.DataFrame(index=self.master_daterange)



    def z_score(self, arr: np.array):
        """
        doc string
        """
        z_score_arr = (arr - np.mean(arr)) / np.std(arr)

        return z_score_arr



    def retrieve_data(self):
        """
        doc string
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