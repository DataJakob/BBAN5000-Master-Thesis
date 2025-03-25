import yfinance as yf
import pandas as pd
import numpy as np
import itertools



class DataRetriever():
    """
    Class to retrieve financial data for given sectors and stock tickers from yf API.

    Attributes:
        sector_df (pd.DataFrame): DataFrame containing sector data with stock tickers.
        start_date (str): The start date for retrieving stock data in 'YYYY-MM-DD' format.
        end_date (str): The end date for retrieving stock data in 'YYYY-MM-DD' format.
        raw_data (List[List[np.ndarray]]): A list of lists containing stock data for each sector.
        returns (List[List[np.ndarray]]): A list of lists containing stock returns for each sector.

    Methods:
        __init__(self, sector_df: pd.DataFrame, start_date: str, end_date: str):
            Initializes the DataRetriever instance with sector data and date range.
        
        retrieve_data(self, log: bool = False) -> str:
            Retrieves stock data and calculates returns for each sector.
    """



    def __init__(self, sector_df,  start_date, end_date):
        """
        Initializes the DataRetriever instance.

        Args:
            sector_df (pd.DataFrame): DataFrame containing data with sector columns and stock tickers as cells.
            start_date (str): The start date for retrieving stock data in 'YYYY-MM-DD' format.
            end_date (str): The end date for retrieving stock data in 'YYYY-MM-DD' format.
        """
        self.sector_df =  sector_df
        self.sectors = sector_df.columns
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = []
        self.returns = []
    


    def retrieve_data(self, log=bool):
        """
        Retrieves stock data for each sector, calculates returns, and stores the results.

        Args:
            log (bool, optional): Whether to print progress information during data retrieval. Defaults to False.

        Returns:
            str: A success message indicating that the data was successfully retrieved.
        """
        
        # For each secor
        for sector in range(0, len(self.sectors),1):

            sector_list = []
            sector_list_return = []

            # For each stock in sector
            for stock in range(0, len(self.sector_df[self.sectors[0]]),1):

                # Retrieve stock prices from yahoo finance
                ind_yf_data = yf.download(str(self.sector_df[self.sectors[sector]][stock]), 
                                          start=self.start_date, end=self.end_date)
                # Handling missing data
                ind_yf_data = ind_yf_data.interpolate(method='linear').ffill().bfill()
                double_data = [[float(ind_yf_data["Open"].iloc[i].iloc[0]), float(ind_yf_data["Close"].iloc[i].iloc[0])] for i in range(len(ind_yf_data))]
                flatten_data = np.array(list(itertools.chain(*double_data)))

                sector_list.append(flatten_data)   

                # Transform the stock prices into returns
                ind_yf_data_return = np.diff(flatten_data)  / flatten_data[:-1]
                ind_yf_data_return_final = np.insert(ind_yf_data_return, 0, 0)
                sector_list_return.append(ind_yf_data_return_final)

                # On command: print data availability for each stock
                if log == True:
                    print(self.sector_df[self.sectors[sector]][stock], "with", len(flatten_data), "observations")

            self.raw_data.append(sector_list)
            self.returns.append(sector_list_return)

        prices_array =  [pd.Series(stock) for sector in self.raw_data for stock in sector]
        prices_df = pd.DataFrame(prices_array)
        prices_tdf = prices_df.T
        prices_tdf.to_csv('Data/StockPrices.csv', index=False)

        return_array =  [pd.Series(stock) for sector in self.returns for stock in sector]
        return_df = pd.DataFrame(return_array)
        return_tdf = return_df.T
        return_tdf.to_csv('Data/StockReturns.csv', index=False)
            
        return "--Data retrieved successfully--"