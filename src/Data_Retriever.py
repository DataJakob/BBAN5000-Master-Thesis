import yfinance as yf
import pandas as pd


class DataRetriever():

    def __init__(self, sector_df,  start_date, end_date):
        self.sector_df =  sector_df
        self.sectors = sector_df.columns
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = []
        self.returns = []
    
    def retrieve_data(self):
        for sector in range(0, len(self.sectors),1):

            sector_list = []
            sector_list_return = []

            for stock in range(0, len(self.sector_df[self.sectors[0]]),1):

                # Retrieve stock prices from yahoo finance
                ind_yf_data = yf.download(str(self.sector_df[self.sectors[sector]][stock]), 
                                          start=self.start_date, end=self.end_date)["Close"]
                sector_list.append(ind_yf_data)   

                # Transform the stock prices into returns
                ind_yf_data_return = ind_yf_data.shift(1)/ind_yf_data
                ind_yf_data_return.iloc[0] = 1
                sector_list_return.append(ind_yf_data_return)

            self.raw_data.append(sector_list)
            self.returns.append(sector_list_return)
        return "--Data retrieved successfully--"