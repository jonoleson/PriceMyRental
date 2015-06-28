# PriceMyRental
***

### Jon Oleson's (that's me) capstone project for Galvanize Data Science Immersive (formerly known as Zipfian Academy). 

## Overview

San Francisco has one the hottest rental markets in the nation, and immense variation according to location and amenities. My goal is to accurately predict market prices for units anywhere in the city. I accomplish this by applying Natural Language Processing (NLP), binary space partitioning, and tree-based regression methods to 9 months of SF rental listings from craigslist. 

Test the app for yourself at [PriceMyRental.io](pricemyrental.io). I recommend testing using [real craigslist listings!](https://sfbay.craigslist.org/search/sfc/apa)

## The Data

I was lucky to get access to 9 months of craigslist rental listing data that has been scraped and stored in AWS buckets in JSON files. Parsing this data was tricky. Some days had no data, and many listings had missing values. To see how I collected the data, refer to [get_data.py](/blob/master/code/get_data.py). The code is written to be able to parse data from all 25 metro areas I have access to, though in practice I only parsed the San Francisco data to intentionally limit the scope of the project. 



