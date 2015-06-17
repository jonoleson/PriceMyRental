import pandas as pd
from datetime import timedelta, date
import urllib2


#Function to iterate through dates
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        single_date = start_date + timedelta(n)
        yield single_date.strftime('%Y-%m-%d')

def get_data(start_date, end_date, one_city=False, print_urls=False):
  #Get dataframe of cities
  cities = pd.read_csv('data/ongoing_cities.csv', header = False)

  # Headers for our target dataframe
  cols = ['id', 'heading', 'body', 'price', 'lat', 'long', 'region', 'neighborhood', 
             'beds', 'baths', 'parking', 'washer_dryer']

  #If True, parse only the data for San Francisco
  if one_city==True:
    cities = cities[(cities['city'] == 'San Francisco')]

  if one_city==True:
    cities = cities['city'] == 'San Francisco'

  for i in xrange(len(cities)):
    if one_city==False:
      city = cities.city_code[i]
      state = cities.state_code[i]
    else:
      city = cities.city_code.values[0]
      state = cities.state_code.values[0]
    city_df = pd.DataFrame(columns=cols)

    #Append data to city_df
    for date in daterange(start_date, end_date):
      done_parsing = False
      k=0

      while done_parsing == False:
        url = 'https://s3-us-west-2.amazonaws.com/hoodsjson/%s/%s/%s/%s.html' %(state, city, date, k)
        #If print_urls==True, Print out the url of each json as it's being parsed
        if print_urls==True:
          print url
        #Read json into pandas dataframe
        #Some days may have no data and may throw an error when loaded, so using try/except to control for this
        try: 
          raw_df = pd.read_json(url)
        except urllib2.HTTPError:
          print 'No data this day'
          break

        #Test if loaded df has any data (the last json file of a day, if that day contains data at all,
        #is always valid but empty)
        if len(raw_df) > 0:
          #Filter df for only craigslist data
          condition = raw_df['source'] == 'CRAIG'
          raw_df = raw_df[condition]
          raw_df = raw_df.reset_index()
          del raw_df['index']
          #Parse raw_df into a usable format
          results_df = parse_info(raw_df, cols)
          city_df = city_df.append(results_df)
          k += 1   
        else:
          done_parsing = True

    #Write city_df to a csv inside a 'data' folder. End result should be a csv file for each city
    city_df.to_csv('data/%s_%s.csv' %(city, state), index=False, encoding='utf-8') 


def parse_info(df, cols):
  results_df = pd.DataFrame(columns = cols, index = xrange(len(df)))
  results_df['id'] = df['id']
  results_df['heading'] = df['heading']
  results_df['body'] = df['body']
  results_df['price'] = df['price']
  results_df['date'] = pd.to_datetime(df['timestamp'], unit='s')

  for i in xrange(len(df)):
    if 'lat' and 'long' in df.iloc[i]['location']:
      results_df.ix[i,'lat'] = float(df.iloc[i]['location']['lat'])
      results_df.ix[i,'long'] = float(df.iloc[i]['location']['long'])
    else:
      results_df.ix[i,'lat'] = None
      results_df.ix[i,'long'] = None 

    if 'source_subloc' in df.iloc[i]['annotations']:
      results_df.ix[i, 'region'] = df.iloc[i]['annotations']['source_subloc']

    if 'source_neighborhood' in df.iloc[i]['annotations']:
      results_df.ix[i, 'neighborhood'] = df.iloc[i]['annotations']['source_neighborhood']
    else:
      results_df.ix[i, 'neighborhood'] = None

    if 'bedrooms' in df.iloc[i]['annotations']:
      #A few listings say 'Studio' rather than '0br' in the bedrooms field
      if df.iloc[i]['annotations']['bedrooms'][0] not in '0123456789':
         results_df.ix[i, 'beds'] = 0
      else:
        results_df.ix[i, 'beds'] = int(df.iloc[i]['annotations']['bedrooms'][0])
    else:
      results_df.ix[i, 'beds'] = None

    if 'bathrooms' in df.iloc[i]['annotations']:
      results_df.ix[i, 'baths'] = int(df.iloc[i]['annotations']['bathrooms'][0])
    else:
      results_df.ix[i, 'baths'] = None
    
    if 'street_parking' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 1
    elif 'carport' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 2
    elif 'off_street_parking' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 3
    elif 'attached_garage' in df.iloc[i]['annotations']:
      results_df.ix[i, 'parking'] = 4
    else:
      results_df.ix[i, 'parking'] = 0

    if 'w_d_in_unit' in df.iloc[i]['annotations']:
      results_df.ix[i, 'washer_dryer'] = 1
    else:
      results_df.ix[i, 'washer_dryer'] = 0

  return results_df

def main():
  start_date = date(2014, 9, 29)
  end_date = date(2015, 6, 17)
  get_data(start_date, end_date, one_city=True, print_urls=True)

if __name__=='__main__':
  main()

