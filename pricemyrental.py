import pandas as pd
import numpy as np 
from featurize import featurize_and_save, create_testing_df
from geopy.geocoders import Nominatim

class PriceMyRental(object):

    def __init__(self):
        self.rfr_file = open('models/rfr.pkl', 'rb')
        self.rfr = cPickle.load(rfr_file) 

    def get_attributes(self, description, beds, baths, address, neighborhood, parking, washer_dryer, price):
        self.description         = description
        self.beds                = beds
        self.baths               = baths
        self.neighborhood        = neighborhood
        self.address             = address
        self.parking             = parking
        self.washer_dryer        = washer_dryer
        self.price               = price

    def get_coords(self):
        geolocator = Nominatim()
        location   = geolocator.geocode(self.address)
        self.lat   = location.latitude
        self.lon   = location.longitude

    def build_df(self):
        cols = ['body', 'beds', 'baths',
                'neighborhood', 'lat', 'lon',
                'parking', 'washer_dryer', 'price']
        single_listing_df = pd.DataFrame(columns=cols)

        single_listing_df['body']         = self.description
        single_listing_df['beds']         = self.beds
        single_listing_df['baths']        = self.baths
        single_listing_df['neighborhood'] = self.neighborhood
        single_listing_df['lat']          = self.lat
        single_listing_df['lon']          = self.lon
        single_listing_df['parking']      = self.parking
        single_listing_df['washer_dryer'] = self.washer_dryer
        single_listing_df['price']        = self.price

        self.df = single_listing_df

    def featurize_listing(self):
        self.df = featurize_and_save(self.df, n_topics=4, n_neighbors=10, single_listing=True)

    def predict(self):
        testing_df = create_testing_df(self.df)

        y = int(testing_df.pop('price').values)
        X = testing_df.values

        prediction = self.rfr.predict(X)

        print 'Your model-recommended price is: %s' %prediction
        print 'Your initial stated price was: %s' %y




