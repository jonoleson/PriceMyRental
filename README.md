# PriceMyRental
***

### Jon Oleson's (that's me) capstone project for Galvanize Data Science Immersive (formerly known as Zipfian Academy). 

## Overview

San Francisco has one the hottest rental markets in the nation, and immense variation according to location and amenities. My goal is to accurately predict market prices for units anywhere in the city. I accomplish this by applying Natural Language Processing (NLP), binary space partitioning, and tree-based regression methods to 9 months of SF rental listings. 

For instructions on using the web app, [go here](../tree/master/webapp).

## The Data

I was lucky to get access to 9 months (September 2014-June 2015) of housing rental data for the top 25 metro areas in the US. The data had been scraped and stored in AWS buckets in JSON files. Parsing this data was tricky; some days had no data, and many listings had missing values. To see how I collected the data, refer to [get_data.py](../blob/master/code/get_data.py). The code is written to be able to parse data from all 25 metro areas I have access to, though in practice I only parsed the San Francisco data to intentionally limit the scope of the project. For each listing, I collected the listing ID, location (in coordinates), neighborhood, number of beds, number of baths, the listing description, and parking information.

Next, I cleaned and de-duplicated the data. I started filtering for data with reliable neighborhood attributes. Then I dropped duplicates according to posting ID and body description text. This alone reduced the rows in the dataset from over 200k to around 75k. To review this code, refer to [clean_data.py](../blob/master/code/clean_data.py). 

## Feature Engineering 

Feature engineering was by far the most crucial element of this project, allowing me to push the accuracy of the prediction model farther than what a model built on only the base numeric features of the data could accomplish. Getting an accurate market price for a rental listing involves determining a few key factors: 

* The median price of an average-sized unit (in our case, a 1-bedroom) in the neighborhood at large 
* The median price of the geographically closest comparables (same number of beds and baths) 
* Whatever information can be extracted from the listing description that may indicate a higher or lower level of quality 

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
6. 4 latent features extracted from description text

Take a look at [featurize.py](../blob/master/code/featurize.py) to review the code for this section. 

## Model Selection

To select the regression model, I ran a grid-search on parameters for a random forest regressor and a ridge regressor. The random forest model yielded superior performance, and was the model I used going forward. I compared performance of each layer of feature engineering and modeling to my "baseline" model which used a random forest regressor predicting off of the base-level dataset, which included neighborhood-level price medians, but lacked latent textual features and the nearest-comparables median price. I also tested performance of only using nearest-comparables median price (the "neighbors median" model) as a second baseline. Mean Absolute Percent Error  was calculated using a 70/30 train-test split, except the standalone neighbors median, which needed no training.

The performance breakdown was as follows:

| Model        |  Mean Absolute Error  |Mean Absolute Percent Error    | R^2 |
| ------------- |:---------------------:|:---------------------------:|:-----:|
| Standalone random forest regressor|   600.8  | 19.5% | 0.730 |
| Standalone neighbors median|    450.9  | 15.4% |  0.784    |
| Random forest regressor + NMF latent features| 467.7| 15.4% |   0.807    |
| RF regressor + latent features + neighbors median| 379.8|  12.7% |  0.859    |

See the code for this section in [grid_search.py](../blob/master/code/grid_search.py) and [models.py](../blob/master/code/models.py), although the code for running the standalone neighbors median model is in [featurize.py](../blob/master/code/featurize.py). 

#### Market Trend Adjustment
It's an obvious concern when using rental data, particularly in San Francisco, that a model trained on past data would produce less valid predictions over time. With that in mind, I did add a seasonal adjustment feature to the final dataset and ran it through my final model to see if I would get improved results. The adjustment consisted of a 'month' term, 0-8, meaning what month the data originated from. To my surprise, adding this term produced no noticeable change in the performance of the final model, so I left it out of my final dataset. As time goes on, however, adding seasonal adjustment, or parsing and training on new data, will certainly be necessary for predictions to remain reliable. 

## The WebApp

I built [PriceMyRental.io](http://pricemyrental.io) on top of the ["Stylish Portfolio"](http://startbootstrap.com/template-overviews/stylish-portfolio/) Bootstrap theme. 

To review the backend code for the app, refer to [app.py](../blob/master/webapp/app.py) and [pricemyrental.py](../blob/master/webapp/pricemyrental.py), which contain the code that runs the app, and makes a prediction based on user input, respectively. 

## Libraries Used

* Numpy
* Pandas
* scikit-learn
* Geopy
* NLTK
