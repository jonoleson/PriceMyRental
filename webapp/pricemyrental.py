import pandas as pd
import numpy as np 
from featurize import featurize_single_listing, create_testing_df
from geopy.geocoders import Nominatim
import cPickle

class PriceMyRental(object):

    def __init__(self, rfr):
        '''
        Initializes the PriceMyRental (PMR) object 
        '''
        self.description  = None
        self.beds         = None
        self.baths        = None
        self.neighborhood = None
        self.address      = None
        self.parking      = None
        self.price        = None
        self.lat          = None
        self.lon          = None
        self.df           = None
        self.rfr          = rfr 

    def get_attributes(self, description, beds, baths, address, 
                       neighborhood, parking, price):
        '''
        Assigns the user-inputted listing attributes to the 
        PMR object
        '''
        self.description  = description
        self.beds         = beds
        self.baths        = baths
        self.neighborhood = neighborhood
        self.address      = address
        self.parking      = parking
        self.price        = price

    def get_coords(self):
        '''
        Extracts coordinates (lat/long) from a user-inputted
        address and assigns it to the PMR object 
        '''
        geolocator = Nominatim()
        location   = geolocator.geocode(self.address)
        self.lat   = location.latitude
        self.lon   = location.longitude

    def build_df(self):
        '''
        Builds the single listing dataframe
        '''
        cols = ['beds', 'baths', 'neighborhood', 'lat', 'lon',
                'parking', 'body', 'price']

        single_listing_df = pd.DataFrame(columns=cols, index=xrange(1))

        single_listing_df['beds']         = self.beds
        single_listing_df['baths']        = self.baths
        single_listing_df['neighborhood'] = self.neighborhood
        single_listing_df['lat']          = self.lat
        single_listing_df['lon']          = self.lon
        single_listing_df['parking']      = self.parking
        single_listing_df['body']         = self.description
        single_listing_df['price']        = self.price

        self.df = single_listing_df

    def featurize_listing(self, kd, search_df, nhood_medians, vectorizer, nmf):
        '''
        Featurizes the single listing dataframe with latent features, 
        neighborhood-level median, and nearest-neighbor median
        '''
        self.df = featurize_single_listing(self.df, kd, search_df, 
                                           nhood_medians, vectorizer, nmf)

    def predict(self):
        '''
        Output a predicted price, and compare to the initial listed of
        user-guessed price
        '''
        testing_df = create_testing_df(self.df)

        y = int(testing_df.pop('price').values)
        X = testing_df.values

        prediction = int(self.rfr.predict(X)[0])

        predict_statement = 'Your model-recommended price is: $%s' %prediction
        compare_statement = 'Your initial stated price was: $%s' %y

        return predict_statement, compare_statement

def load_data_and_models():
    '''
    Load the data and models that will be used to build the PMR object
    '''
    rfr_file = open('models/rfr.pkl', 'rb')
    rfr = cPickle.load(rfr_file) 

    #Load search_df to be used in finding median neighbors price for a single listing.
    search_df = pd.read_csv('data/search_df.csv', header=False)

    #Create dictionary that contains each neighborhood and the median rent
    #of a 1-bedroom in that neighborhood
    hoods_grouped = search_df.groupby('neighborhood').median()
    nhood_medians = {}

    for hood in hoods_grouped.index:
        nhood_medians[hood] = hoods_grouped.ix[hood]['price_1bd_med']

    #Unpickle the tfidf vectorizer to vectorize the description text
    vect_file  = open('models/tfidf.pkl', 'rb')
    vectorizer = cPickle.load(vect_file)

    #Unpickle the NMF model to generate latent feature weights 
    nmf_file = open('models/nmf.pkl', 'rb')
    nmf      = cPickle.load(nmf_file)

    #Unpickle the KD-Tree to get median neighbor price
    kd_file   = open('models/kd_tree.pkl', 'rb')
    kd        = cPickle.load(kd_file)

    return rfr, search_df, nhood_medians, vectorizer, nmf, kd 

def run_pmr(beds, baths, address, neighborhood, parking, 
            description, price, rfr, search_df, 
            nhood_medians, vectorizer, nmf, kd):
    '''
    Build the PMR object and output a prediction 
    '''
    pmr = PriceMyRental(rfr)
    pmr.get_attributes(description, beds, baths, address, 
                       neighborhood, parking, price)
    pmr.get_coords()
    pmr.build_df()
    pmr.featurize_listing(kd, search_df, nhood_medians, vectorizer, nmf)

    predict_statement, compare_statement = pmr.predict()

    return predict_statement, compare_statement



