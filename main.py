import requests
from dotenv import load_dotenv
import os
import pandas as pd
from http import HTTPStatus

load_dotenv()
api_key = os.getenv("API_KEY")


# Testing things

"""
ticker_symbol = "GOOGL"
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker_symbol}&apikey={api_key}'

r = requests.get(url)
#=print(r.status_code)
data = r.json()
print(data.keys())
"""
"""
Data is in this format: 

data["Meta Data"] gives info on info given, stock symbol, last refreshed date, etc.
data["Time Series (Daily)"] gives daily stock info in nested dict format with date as keys and another dict as value with keys:
    "1. open", "2. high", "3. low", "4. close", "5. volume"
Ex: data["Time Series (Daily)"]['2025-12-15'] gives dict with keys ('1. open', '2. high', '3. low', '4. close', '5. volume') and their respective values for that date.
"""

#df = pd.DataFrame(data['Time Series (Daily)'].keys())
#df = pd.concat([df, pd.DataFrame([data['Time Series (Daily)'][key] for key in data['Time Series (Daily)'].keys()])], axis=1)
#f.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

#df = pd.read_csv("/Users/phoebuschen/VS Code Projects/Finance Models/daily_IBM.csv")
#print(df.head())

#output_file = f"daily_{ticker_symbol}.csv"
#data.to_csv(output_file)
#print(f"Data saved to {output_file}")



def save_data(data):
    """
    Saves stock data to a CSV file named daily_<TICKER_SYMBOL>.csv. If the file already exists, it does not overwrite it.
    
    Params: data - a dict of stock prices in the format returned by the Alpha Vantage API.
    """
    ticker_symbol = data['Meta Data']['2. Symbol']

    df = pd.DataFrame(data['Time Series (Daily)'].keys())
    df = pd.concat([df, pd.DataFrame([data['Time Series (Daily)'][key] for key in data['Time Series (Daily)'].keys()])], axis=1)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    output_file = f"daily_{ticker_symbol}.csv"
    if(not os.path.exists(output_file)):
        df.to_csv(output_file)
        print(f"Data saved to {output_file}")
    else:
        print(f"File {output_file} already exists. Data not saved to avoid overwriting.")

def save_multi_data(ticker_list):
    """
    Saves stock data for multiple ticker symbols to CSV files named daily_<TICKER_SYMBOL>.csv. If a file already exists, it does not overwrite it.
    Params: ticker_list - a list of ticker symbols (strings).
    """
    for symbol in ticker_list:
        output_file = f"daily_{symbol}.csv"

        if(os.path.exists(output_file)):
            print(f"File {output_file} already exists. Skipping download to avoid overwriting.")
            continue

        else:
            print(symbol)
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
            r = requests.get(url)
            data = r.json() 
            if r.status_code == HTTPStatus.OK:
                if("Error Message" in data.keys()):
                    print(f"Error retrieving data for {symbol}: {data['Error Message']}")
                elif("Information" in data.keys()):
                    print(f"API call frequency exceeded. Please wait and try again later.\nError: {data['Information']}")
                    return
                else:  
                    print(data.keys())
                    save_data(data)
            else:
                print(f"Failed to retrieve data for {symbol}. Status code: {r.status_code}")
            

save_multi_data(['APPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])