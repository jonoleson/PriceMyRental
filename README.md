# PriceMyRental
***

### Jon Oleson's (that's me) capstone project for Galvanize Data Science Immersive (formerly known as Zipfian Academy). 

## Overview

San Francisco has one the hottest rental markets in the nation, and immense variation according to location and amenities. My goal is to accurately predict market prices for units anywhere in the city. I accomplish this by applying Natural Language Processing (NLP), binary space partitioning, and tree-based regression methods to 9 months of SF rental listings from craigslist. 

Test the app for yourself at [PriceMyRental.io](pricemyrental.io). I recommend testing using [real craigslist listings!](https://sfbay.craigslist.org/search/sfc/apa)

## The Data

I was lucky to get access to 9 months of craigslist rental listing data that has been scraped and stored in AWS buckets in JSON files. Parsing this data was tricky; some days had no data, and many listings had missing values. To see how I collected the data, refer to [get_data.py](/blob/master/code/get_data.py). The code is written to be able to parse data from all 25 metro areas I have access to, though in practice I only parsed the San Francisco data to intentionally limit the scope of the project. For each listing, I collected the listing ID, location (in coordinates), neighborhood, number of beds, number of baths, the listing description, and parking information.

Next, I cleaned and de-duplicated the data. I started filtering for data with reliable neighborhood attributes (craigslist standarizes this for San Francisco listings but some posters don't use the pre-set neighborhood option). Then I dropped duplicates according to posting ID and body description text. This alone reduced the rows in the dataset from over 200k to around 75k. To review this code, refer to [clean_data.py](/blob/master/code/clean_data.py). 

## Feature Engineering 

Feature engineering was by far the most crucial element of this project, allowing me to push the accuracy of the prediction model farther than what a model built on only the base numeric features of the data could accomplish. Getting an accurate market price for a rental listing involves determining a few key factors: The median price of an average-sized unit (in our case, a 1-bedroom) in the neighborhood at large, the median price of the geographically closest comparables (same number of beds and baths), and whatever information can be extracted from the listing description that may indicate a higher or lower level of quality. 

#### Nearest-Neighbor Search with KD-Trees
Finding the geographically closest comparable listings to any given listing proved to be no trivial task. A brute force method, which would involve calculating distances between each listing and every other listing in the dataset, would be completely computationally infeasible. I had to find a more efficient method, and found it with [KD-Trees](https://www.youtube.com/watch?v=TLxWtXEbtFE). I used a KD-Tree to recursively partition my dataset by latitude and longitude, then searching for [nearest neighbors](https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search) using the tree structure to eliminate large sections of the search space. 

For each listing, I found the geographically closest listings (350, so that I could be reasonably sure there would be enough comparables) and filtered these for comparables, which I define to be listings with the same number of beds and baths as the search listing. I then took the median price of the 10 closest comparables of each listing, and added it as a feature in the dataset. 

#### TF-IDF + NMF to Quantify Latent Features
I further featurized the dataset by vectorizing each listing's description text with TF-IDF weighting, then extracting a matrix of latent feature weights using NMF. I set the NMF model to uncover 4 latent features from the description text, yielding a nx4 matrix, where n is the number of listings in my dataset. I then concatenated this latent feature matrix with the original dataset.  

#### Final Dataset
After feature engineering and subsetting for only features used in the final regression model, the dataset consisted of the following features:
1. Number of beds
2. Number of baths
3. Parking amenity (ranked 0-4 depending on type of parking amenity)
4. Neighborhood-wide median price of a 1-bedroom listing
5. Median price of 10 geographically closest comparables (same number of beds/baths)
6-9. 4 latent features extracted from description text

Take a look at [featurize.py](/blob/master/code/featurize.py) to review the code for this section. 

## Model Selection

To select the regression model, I ran a grid-search on parameters for a random forest regressor and a ridge regressor. The random forest model yielded superior performance, and was the model I used going forward. I compared performance of each layer of feature engineering and modeling to my "baseline" model which used a random forest regressor predicting off of the base-level dataset, which included neighborhood-level price medians, but lacked latent textual features and the nearest-comparables median price. I also tested performance of only using nearest-comparables median price (the "neighbors median" model) as a second baseline. R^2 was calculated using a 70/30 train-test split, except the standalone neighbors median, which needed no training.

The performance breakdown was as follows:

| Model        | R^2          | 
| ------------- |:-------------:| 
| Standalone random forest regressor     | 0.730 | 
| Standalone neighbors median      | 0.784 |   
| Random forest regressor + NMF latent features | 0.807 |    
| RF regressor + latent features + neighbors median| 0.856 |

See the code for this section in [grid_search.py](/blob/master/code/grid_search.py) and [models.py](/blob/master/code/models.py), although the code for running the standalone neighbors median model is in [featurize.py](/blob/master/code/featurize.py). 

## The WebApp

I built [PriceMyRental.io](pricemyrental.io) on top of the ["Stylish Portfolio"](http://startbootstrap.com/template-overviews/stylish-portfolio/) Bootstrap theme. 

To review the backend code for the app, refer to [app.py](blob/master/webapp/app.py) and [pricemyrental.py](blob/master/webapp/pricemyrental.py), which contain the code that runs the app, and makes a prediction based on user input, respectively. 
