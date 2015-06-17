import pandas as pd 

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
sfdf.sort('date')
sfdf.drop_duplicates('id')
sfdf.drop_duplicates('body')
#This should have reduced around 200k datapoints into roughly 81k 

