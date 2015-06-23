import pandas as pd
import numpy as np 
from geopy.geocoders import Nomanatim

class PriceMyRental(object):

    def __init__(self, beds, baths, address, neighborhood, parking, washer_dryer):
        self.neighborhood_median=None 
        self.neighbors_median=None

        


