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

  headers = ["id", "header", "body", "bedrooms", "bathrooms", "sqft", 
             "timestamp", "price", "external_url", "lat", "lon", 
             "accuracy", "address", "parking", "washer_dryer", "pet"]   

  for row in cities:
    city = row.city_code
    state = row.state_code
    city_df = pd.DataFrame(columns=headers)

    #Append data to city_df
    for date in daterange(start_date, end_date):
      i=0
      while i:
        url = 'https://s3-us-west-2.amazonaws.com/hoodsjson/%s/%s/%s/%s.html' %(state, city, date, i)
        r = requests.get(url)
        data_hash = r.json() 
        if len(data_hash) > 0:
          results = parse_info(data_hash)
          results_df = pd.DataFrame(results)
          city_df = pd.concat(city_df, results_df)
          i=+1   
        else:
          break

    #Write city_df to a csv 
    city_df.to_csv('csvs/%s_%s.csv', 'wb+') %(city, state)

def parse_info(data_hash, headers):
  results_df = pd.DataFrame(columns = headers)
  for apt in data_hash:
    header = apt_info['heading']
    body = apt_info['body']
    timestamp = Time.at(apt_info['timestamp'].to_i).utc.to_datetime.to_s[0..9]  
    price = apt_info['price']
    external_url = apt_info['external_url']
    lat = apt_info['location']['lat']
    lon = apt_info['location']['long']
    accuracy = apt_info['location']['accuracy']
    address = apt_info['location']['formatted_address']
    rooms = apt_info['annotations']['bedrooms']
    parking = apt_info['annotations']['carport']
    washer_dryer = apt_info['annotations']['w_d_in_unit']
    


