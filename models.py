import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
  



"""
Given a file path to futures data, returns a formatted DataFrame. 
"""
def clean_soybean_data(path, col_names):
        #read dataframe
        futures = pd.read_csv(path, sep="\t", header=None)
        futures.columns = col_names
        futures['date'] = pd.to_datetime(futures['date'])
        #make numeric
        for col in col_names:
                if col != 'date':
                        # df['colname'] = df['colname'].str.replace(',', '').astype(float)
                        futures[col] = futures[col].str.replace(',', '') #.astype(float) 
                        futures = futures.where( futures != "-", np.nan)
                        futures[col] = pd.to_numeric(futures[col])
        # print(futures.head)
        return futures



"""
This function matches input dataframe columns on date with covid data, showing 
trends over the input period value.
@params: 
    futures_df: dataframe input to match covid data for
    periods: num periods of covid trends 
returns:
    two dataframs, covid trends and corresponding futures prices. NOTE: data contains nan
"""
def make_data(futures_raw ,period): #TODO  CHANGE THIS 
    covid_raw = pd.read_csv("./data/covid_data/covid_us.csv")
    #convert date to datetime format
    covid_raw['date'] = pd.to_datetime(covid_raw['date'])
    #transform data from levels to percent changes
    cases_shift = covid_raw['cases'] #.shift(periods= 1) 
    deaths_shift = covid_raw['deaths'] #.shift(periods= 1) 
    ## OVERWRITING ORIGINALS
    covid_raw['daily_cases'] = cases_shift.diff(periods= period) 
    covid_raw['daily_deaths'] = deaths_shift.diff(periods= period) 
    covid_raw['cases_FD'] = covid_raw['daily_cases'].subtract(covid_raw['daily_cases'].shift(1), axis=0)
    covid_raw['deaths_FD'] = covid_raw['daily_deaths'].subtract(covid_raw['daily_deaths'].shift(1), axis=0)
    print(covid_raw.mean(axis=0))
    futures_raw['daily_close'] = futures_raw['close'].diff(periods= -period) 
    futures_raw['price_FD'] = futures_raw['daily_close'].subtract(futures_raw['daily_close'].shift(-1), axis=0)

    #merge on date
    merged = covid_raw.merge(futures_raw, how='right', on='date')

    merged.date = pd.to_datetime(merged.date, format= "%y-%m-%d")
    # replace infinities
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    #forward fill null values
    merged= merged.fillna(method="ffill",axis=0) 
    #covert to time
    # merged['date'] = pd.to_datetime(merged['date'])
    # reproduce two sets, now matching on date 
    # covid_df = merged[['date','cases', 'deaths']]
    # dates_df = merged['date']
    # futures_df = merged[['date','open', 'high', 'low', 'close', 'adj_close', 'volume']]

    return merged

