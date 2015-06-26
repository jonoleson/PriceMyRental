import pandas as pd 
import numpy as np 
import datetime 

def clean_data():
    # Load the csv saved by get_data
    raw_df = pd.read_csv('data/USA-SFO-SNF_USA-CA.csv', header = False)

    # Filter the data for listings from the City of San Francisco specifically
    sfdf = raw_df[(raw_df['region']=='sfc')]

    #Get list of top neighborhoods in San Francisco
    nhoods = list((sfdf.neighborhood.value_counts()[:39]).keys())

    #Remove erroneous neighborhood labels
    nhoods.remove('San Francisco')
    nhoods.remove('San Francisco, CA')
    nhoods.remove('all neighborhoods')

    #Filter for only the neighborhoods that craigslist explicitly codifies for in SF
    condition = sfdf['neighborhood'].isin(nhoods)
    sfdf = sfdf[condition]

    #Sort dataframe by date, then de-duplicate according to listing IDs and body text, 
    #keeping the most recent listing
    sfdf = sfdf.sort('date')
    sfdf = sfdf.drop_duplicates('id')
    sfdf = sfdf.drop_duplicates('body')

    #Reset the index, and delete the old index column created by this operation
    sfdf = sfdf.reset_index()
    del sfdf['index']

    #Drop listings which still contain NA values, which now should only occur in 
    #the 'beds' and 'baths' columns. These listings tend to be unreliable and misleading. 
    sfdf = sfdf.dropna(axis=0)

    #This should have reduced around 200k datapoints into roughly 75k 

    #Remove outliers filtering for only units priced between 500 and 20k/month
    sfdf = sfdf[(sfdf['price'] < 20000)]
    sfdf = sfdf[(sfdf['price'] > 500)]

    #Add a month column containing the month and year from our 9-month interval
    #so we can graph price change over time
    yrmonths = [x[:7] for x in sfdf['date']]
    sfdf['year-month'] = yrmonths

    return sfdf 

