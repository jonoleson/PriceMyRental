import pandas as pd 
import matplotlib.pyplot as plt 
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
    condition = (sfdf['neighborhood'].isin(nhoods)
    sfdf = sfdf[condition]

    #Sort dataframe by date, then de-duplicate according to listing IDs and body text, 
    #keeping the most recent listing
    sfdf = sfdf.sort('date')
    sfdf = sfdf.drop_duplicates('id')
    sfdf = sfdf.drop_duplicates('body')
    #Drop listings which still contain NA values, which now should only occur in 
    #the 'beds' and 'baths' columns. These listings tend to be unreliable and misleading. 
    sfdf = sfdf.dropna(axis=0)

    #This should have reduced around 200k datapoints into roughly 75k 

    #Remove obvious outliers and listings with NaNs in price category
    sfdf = sfdf['price'].dropna()
    #Somewhat arbitrarily filtering for only units priced at under 20k/month
    #Properties listed at over 20k/month are either extreme luxury outliers or typos
    sfdf = sfdf[(sfdf['price'] < 20000)]

    #Create df containing nighborhood median rents for 1-bedroom apartments
    sfdf_grouped = sfdf[(sfdf['beds']==1)].groupby('neighborhood').median()

    #Set original df's index to the neighborhood field
    sfdf.set_index('neighborhood', inplace=True)

    #Join sfdf and sfdf_grouped to add the average 1_bedroom price for a given 
    #neighborhood to sfdf
    sfdf = sfdf.join(sfdf_grouped[['price']], rsuffix='_1bd_med') 

    #Add a month column containing the month and year from our 9-month interval
    #so we can graph price change over time
    yrmonths = [x[:7] for x in sfdf['date']]
    sfdf['year-month'] = yrmonths

    #Subset dataframe into only features that will be used in testing
    test_df = sfdf[['beds', 'baths', 'parking', 'washer_dryer', 'price_1bd_med', 'price']]

    return sfdf, test_df

def graph_trend(sfdf):
    df_copy = sfdf.copy()
    df_grouped_median = df_copy[(df_copy['beds']==1)].groupby('year-month').median()
    plt.figure(figsize=(10,10))
    df_grouped_median['price'].plot()