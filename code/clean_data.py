import pandas as pd 
import numpy as np 
import datetime 


def load_data():
    '''
    Input: None
    Output: Pandas dataframe loaded from CSV saved in get_data.py 
    '''
    raw_df = pd.read_csv('../data/USA-SFO-SNF_USA-CA.csv', header = False)
    return raw_df


def filter_neighborhoods(raw_df):
    '''
    Input: Pandas dataframe
    Output: Pandas dataframe filtered for listings containing craigslist's correct 
    SF neighborhood labels
    '''
    # Filter the data for listings from the City of San Francisco specifically
    sfdf = raw_df[(raw_df['region']=='sfc')]

    # Get list of top neighborhoods in San Francisco
    nhoods = list((sfdf.neighborhood.value_counts()[:39]).keys())

    # Remove erroneous neighborhood labels
    nhoods.remove('San Francisco')
    nhoods.remove('San Francisco, CA')
    nhoods.remove('all neighborhoods')

    # Filter for only the neighborhoods that craigslist explicitly 
    # codifies for in SF
    condition = sfdf['neighborhood'].isin(nhoods)
    sfdf = sfdf[condition]

    return sfdf


def remove_dupes(sfdf):
    '''
    Input: Pandas dataframe
    Output: De-duplicated Pandas dataframe, keeping the most recent 
    duplicate of any listing
    '''
    # Sort dataframe by date, then de-duplicate according to  
    # listing IDs and body text, keeping the most recent listing
    sfdf = sfdf.sort('date')
    sfdf = sfdf.drop_duplicates('id')
    sfdf = sfdf.drop_duplicates('body')

    # Reset the index, and delete the old index column created by this operation
    sfdf = sfdf.reset_index()
    del sfdf['index']

    # Drop listings which still contain NA values, which now should 
    # only occur in the 'beds' and 'baths' columns. 
    # These listings tend to be unreliable and misleading. 
    sfdf = sfdf.dropna(axis=0)

    # This should have reduced around 200k datapoints into roughly 75k 
    return sfdf


def remove_outliers(sfdf):
    '''
    Input: Pandas dataframe
    Output: Pandas dataframe filtered for only units priced between 500 
    and 20k/month
    '''
    sfdf = sfdf[(sfdf['price'] < 20000)]
    sfdf = sfdf[(sfdf['price'] > 500)]

    return sfdf


def get_year_month(sfdf):
    '''
    Input: Pandas dataframe
    Output: Pandas dataframe with year-month column allowing us to 
    track price trends over time
    '''
    yrmonths = [x[:7] for x in sfdf['date']]
    sfdf['year-month'] = yrmonths

    return sfdf
  

def save_df(sfdf):
    '''
    Input: Fully-cleaned pandas dataframe
    Output: CSV file, saved to the data folder
    '''
    sfdf.to_csv('../data/sf_clean.csv', index=False, encoding='utf-8')


def main():
    raw_df = load_data()
    sfdf   = filter_neighborhoods(raw_df)
    sfdf   = remove_dupes(sfdf)
    sfdf   = remove_outliers(sfdf)
    sfdf   = get_year_month(sfdf)
    save_df(sfdf)

if __name__=='__main__':
    main()

