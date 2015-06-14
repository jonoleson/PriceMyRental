import pandas as pd
import csv
from datetime import timedelta, date
import requests

start_date = date(2014, 9, 28)
end_date = date(2014, 10, 28)

#Function to iterate through dates
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        single_date = start_date + timedelta(n)
        yield single_date.strftime('%Y-%m-%d')

def get_data():
  #Get dataframe of cities
  cities = pd.read_csv('ongoing_cities.csv', headers = True)

  ''' 
  Target headers will look like:
  headers = ["id", "header", "body", "bedrooms", "bathrooms", "sqft", 
             "timestamp", "price", "external_url", "lat", "lon", 
             "accuracy", "address", "parking", "washer_dryer", "pet"]   
  '''
  # Headers for the dataframes loaded by pd.read_json()
  headers = ['annotations', 'body', 'category', 'expires', 'external_url', 'flagged_status', 
             'heading', 'id', 'immortal', 'location', 'price', 'source', 'timestamp']

  for row in cities:
    city = row.city_code
    state = row.state_code
    city_df = pd.DataFrame(columns=headers)

    #Append data to city_df
    for date in daterange(start_date, end_date):
      done_parsing = False

      while done_parsing = False:
        url = 'https://s3-us-west-2.amazonaws.com/hoodsjson/%s/%s/%s/%s.html' %(state, city, date, i)
        
        #Read json into pandas dataframe
        raw_df = pd.read_json(url)
        #Filter df for only craigslist data
        condition = raw_df['source'] == 'CRAIG'
        raw_df = raw_df[condition]

        #Test if loaded df has any data
        if len(raw_df) > 0:
          results = parse_info(raw_df, headers)
          results_df = pd.DataFrame(results)
          city_df = pd.concat(city_df, results_df)
          i=+1   
        else:
          done_parsing = True

    #Write city_df to a csv 
    city_df.to_csv('csvs/%s_%s.csv', 'wb+') %(city, state)

def parse_info(df, headers):
  results_df = pd.DataFrame(columns = headers, index = xrange(len(df)))
  for i, apt in enumerate(df):
    results_df['header'][i] = apt['heading']
    results_df['body'][i] = apt['body']
    results_df['price'][i] = apt['price']
    results_df['lat'][i] = apt['location']['lat']
    results_df['lon'][i] = apt['location']['long']
    results_df['accuracy'][i] = apt['location']['accuracy']
    results_df['address'][i] = apt['location']['formatted_address']
    results_df['beds'][i] = apt['annotations']['bedrooms']
    results_df['baths'][i] = apt['annotations']['bathrooms']
    
    if 'street_parking' in apt['annotations'][0]:
      results_df['parking'][i] = 1
    elif 'carport' in apt['annotations'][0]:
      results_df['parking'][i] = 2
    elif 'off_street_parking' in apt['annotations'][0]:
      results_df['parking'][i] = 3
    elif 'attached_garage' in apt['annotations'][0]:
      results_df['parking'][i] = 4
    else:
      results_df['parking'][i] = 0

    results_df['parking'][i] = apt['annotations']['carport']
    results_df['washer_dryer'][i] = apt['annotations']['w_d_in_unit']

  results_df['date'] = pd.to_datetime(df['timestamp'], unit='s')
  return results_df


