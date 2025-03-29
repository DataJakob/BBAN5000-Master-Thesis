import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler



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



    def __init__(self, sector_df,  start_date, end_date, history_usage_RL):
        """
        Initializes the DataRetriever instance.

        Args:
            sector_df (pd.DataFrame): DataFrame containing data with sector columns and stock tickers as cells.
            start_date (str): The start date for retrieving stock data in 'YYYY-MM-DD' format.
            end_date (str): The end date for retrieving stock data in 'YYYY-MM-DD' format.
        """
        self.sector_df: pd.DataFrame =  sector_df
        self.sectors: list = sector_df.columns
        self.history_usage: int = history_usage_RL
        self.start_date: str = start_date
        self.end_date: list = end_date
        self.raw_data: list = []
        self.returns: list = []
        self.volume:list = []
        self.rolling_return: list = []
        self.rolling_std: list = []
    


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
            volume_list = []
            rolling_return_list = []
            rolling_volatility_list = []

            # For each stock in sector
            for stock in range(0, len(self.sector_df[self.sectors[0]]),1):

                # Retrieve stock prices from yahoo finance
                ind_yf_data = yf.download(str(self.sector_df[self.sectors[sector]][stock]), 
                                          start=self.start_date, end=self.end_date)
                # Handling missing data
                ind_yf_data = ind_yf_data.interpolate(method='linear').ffill().bfill()

                # Choose both open and close data and zip them togehter
                double_data = [[float(ind_yf_data["Open"].iloc[i].iloc[0]), float(ind_yf_data["Close"].iloc[i].iloc[0])] for i in range(len(ind_yf_data))]
                
                flatten_data = np.array(list(itertools.chain(*double_data)))
                sector_list.append(flatten_data)   

                # Add volume data
                volume_data = np.repeat(ind_yf_data["Volume"].values, 2).reshape(-1,1)
                scaler = StandardScaler()
                volume_list.append(scaler.fit_transform(volume_data))

                # Transform the stock prices into returns
                ind_yf_data_return = np.diff(flatten_data)  / flatten_data[:-1]
                ind_yf_data_return_final = np.insert(ind_yf_data_return, 0, 0)
                sector_list_return.append(ind_yf_data_return_final)

                # Rolling return and volatility
                first_return = []
                first_volatility = []
                for i in range(1,10,1):
                    ind_returns = np.prod(ind_yf_data_return_final[:i]+1)-1
                    ind_vol = np.std(ind_yf_data_return_final[:i])
                    first_return.append(ind_returns)
                    first_volatility.append(ind_vol)
                total_return = (pd.Series(ind_yf_data_return_final)+1).rolling(10).apply(np.prod, raw=True) -1
                total_return[:9] =  first_return
                total_volatilty = pd.Series(ind_yf_data_return_final).rolling(10).std() 
                total_volatilty[:9] = first_volatility
                rolling_return_list.append(total_return)
                rolling_volatility_list.append(total_volatilty)

                # On command: print data availability for each stock
                if log == True:
                    print(self.sector_df[self.sectors[sector]][stock], "with", len(flatten_data), "observations")

            self.raw_data.append(sector_list)
            self.returns.append(sector_list_return)
            self.volume.append(volume_list)
            self.rolling_return.append(rolling_return_list)
            self.rolling_std.append(rolling_volatility_list)

        prices_array =  [pd.Series(stock) for sector in self.raw_data for stock in sector]
        prices_df = pd.DataFrame(prices_array)
        prices_tdf = prices_df.T
        prices_tdf.to_csv('Data/Input/StockPrices.csv', index=False)

        return_array =  [pd.Series(stock) for sector in self.returns for stock in sector]
        return_df = pd.DataFrame(return_array)
        return_tdf = return_df.T
        return_tdf.to_csv('Data/Input/StockReturns.csv', index=False)

        volume_array = [pd.Series(stock.flatten()) for sector in self.volume for stock in sector]
        volume_df = pd.DataFrame(volume_array)
        volume_dft = volume_df.T
        volume_dft.to_csv("Data/Input/StockVolume.csv", index=False)
            
        rollret_array = [pd.Series(stock) for sector in self.rolling_return for stock in sector]
        rollret_df = pd.DataFrame(rollret_array)
        rollret_dft = rollret_df.T
        rollret_dft.to_csv("Data/Input/RollingRet.csv", index=False)

        rollvol_array = [pd.Series(stock) for sector in self.rolling_std for stock in sector]
        rollvol_df = pd.DataFrame(rollvol_array)
        rollvol_dft = rollvol_df.T
        rollvol_dft.to_csv("Data/Input/RollingVol.csv", index=False)

        pd.concat([return_tdf, volume_dft, rollret_dft, rollvol_dft], axis=1).to_csv("Data/Input/Total.csv", index=False)

    
        return "--Data retrieved successfully--"