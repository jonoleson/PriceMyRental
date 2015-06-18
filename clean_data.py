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
sfdf = sfdf.sort('date')
sfdf = sfdf.drop_duplicates('id')
sfdf = sfdf.drop_duplicates('body')

#This should have reduced around 200k datapoints into roughly 81k 

#Remove obvious outliers and listings with NaNs in price category
sfdf = sfdf['price'].dropna()
#Somewhat arbitrarily filtering for only units priced at under 20k/month
sfdf = sfdf[(sfdf['price'] < 20000)]

#Create df containing nighborhood average rents for 1-bedroom apartments
sfdf_grouped = sfdf[(sfdf['beds']==1)].groupby('neighborhood').mean()

#Set original df's index to the neighborhood field
sf_deduped.set_index('neighborhood', inplace=True)

#Join sfdf and sfdf_grouped to add the average 1_bedroom price for a given 
#neighborhood to sfdf
sf_deduped.join(df_grouped[['price']], rsuffix='_1bd_avg') 