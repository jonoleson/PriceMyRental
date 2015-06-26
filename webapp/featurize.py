import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktLanguageVars
import re
from sklearn.decomposition import NMF
from sklearn.neighbors import KDTree 
import cPickle


def create_search_df(df):
    #Create a search_df to be used in finding median neighbors price for a single listing.
    search_df = df[['beds', 'baths','neighborhood', 'price_1bd_med', 'price']]
    search_df.to_csv('data/search_df.csv', index=False, encoding='utf-8')


def get_neighborhood_median(df):
    #Create df containing nighborhood median rents for 1-bedroom apartments
    df_grouped = df[(df['beds']==1)].groupby('neighborhood').median()

    #Set original df's index to the neighborhood field
    df.set_index('neighborhood', inplace=True)

    #Join sfdf and sfdf_grouped to add the average 1_bedroom price for a given 
    #neighborhood to sfdf
    df = df.join(df_grouped[['price']], rsuffix='_1bd_med')

    df.reset_index(inplace=True)
    return df
 

def get_median_neighbors(df, n_neighbors):
    '''
    This function forecasts price by searching for n_neighbors closest 
    listings to each listing and returning the median price of those 
    listings. It does this by using a KD-Tree to efficiently find the
    listings closest to the current listing by location (lat, lon). 
    The indices of these closest listings are then used to find the 
    5 closest listings that have the same layout (same # of beds and baths).
    It returns an RMSE to test its baseline performance as a standalone 
    model and returns a modified DataFrame that can be passed in to 
    another model.
    '''
    kd_df = df[['lat', 'lon']]
    kdvals = kd_df.values
    kd = KDTree(kdvals, leaf_size = 1000)
    cPickle.dump(kd, open('models/kd_tree.pkl', 'wb'))
    neighbors = kd.query(kdvals, k=350)

    median_neighbor_prices = []
    for i in xrange(len(df)):
        listing_neighbors = neighbors[1][i]
        listing_id        = int(df.ix[i,'id'])
        n_beds            = int(df.ix[i,'beds'])
        n_baths           = int(df.ix[i,'baths'])

        sub_df = df[(df.index.isin(listing_neighbors))]
        sub_df = sub_df[
            (sub_df['beds']  == n_beds)  &
            (sub_df['baths'] == n_baths) &
            (sub_df['id']    != listing_id)
            ]

        comp_listings = [item for item in listing_neighbors if item in sub_df.index]
        
        med_price = df.price[comp_listings][:n_neighbors].median()
        median_neighbor_prices.append(med_price)

    df['med_neighbor_price'] = median_neighbor_prices
       
    rmse = np.mean((df['med_neighbor_price'] - df['price'])**2)**0.5
    print 'RMSE is ', rmse
    return df


def clean(list_of_docs):
    clean_docs = []
    for doc in list_of_docs:
        clean_data = re.sub("[^a-zA-Z]"," ", doc) #grabs only letters
        clean_data = re.sub(' +',' ', clean_data) #removes multiple spaces
        clean_data = clean_data.lower() #converts to lower case
        clean_docs.append(clean_data)
    return clean_docs


def tokenize(doc):
    plv = PunktLanguageVars()
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in plv.word_tokenize(doc.lower())]


def get_tfidf(clean_docs):
    vectorizer = TfidfVectorizer('corpus', tokenizer = tokenize,
                                  stop_words=stopwords.words('english'), 
                                  strip_accents='unicode', norm='l2')
    vectorizer.fit(clean_docs)
    cPickle.dump(vectorizer, open('models/tfidf.pkl', 'wb'))
    X = vectorizer.transform(clean_docs)
    return vectorizer, X


def run_nmf(X, vectorizer, n_topics=4, print_top_words=False):
    nmf = NMF(n_components=n_topics)
    nmf.fit(X)
    cPickle.dump(nmf, open('models/nmf.pkl', 'wb'))
    H = nmf.transform(X)

    if print_top_words==True:
        feature_names = vectorizer.get_feature_names()
        n_top_words   = 10    
        for topic_idx, topic in enumerate(nmf.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()

    return H #Return the H matrix


def add_latent_features(df, n_topics):
    clean_text     = clean(df.body.values)
    vectorizer, X  = get_tfidf(clean_text)
    latent_weights = run_nmf(X, vectorizer, n_topics=n_topics)
    latent_df      = pd.DataFrame(latent_weights, columns=(
                                ['Latent Feature %s' % (i+1) for i in range(n_topics)]))
    
    concat_df = pd.concat([df, latent_df], axis=1)
    return concat_df


def featurize_and_save(df, n_topics, n_neighbors):
    df = add_latent_features(df, n_topics=4)

    df = get_median_neighbors(df, n_neighbors=n_neighbors)
    #A few extremely unique listings couldn't find a median price for comparables, 
    #so we're dropping them from the dataset 
    df = df.dropna(axis=0)
    df.to_csv('data/complete_df.csv', index=False, encoding='utf-8')
 

def get_single_listing_median_neighbors(single_df, kd, search_df):
    n_beds  = int(single_df.beds.values)
    n_baths = int(single_df.baths.values)
    lat     = float(single_df.lat.values)
    lon     = float(single_df.lon.values)

    location  = np.array([lat, lon])
    neighbors = kd.query(location, k=1000)

    listing_neighbors = neighbors[1][0]

    sub_df = search_df[(search_df.index.isin(listing_neighbors))]
    sub_df = sub_df[
        (sub_df['beds']  == n_beds)  &
        (sub_df['baths'] == n_baths) 
        ]

    comp_listings = [item for item in listing_neighbors if item in sub_df.index]
    med_price = search_df.price[comp_listings][:10].median()
    single_df['med_neighbor_price'] = med_price
    return single_df   


def run_nmf_single(X, nmf):
    H = nmf.transform(X)
    return H


def add_latent_features_single(df, vectorizer, nmf):
    clean_text     = clean(df.body.values)
    X              = vectorizer.transform(clean_text)
    latent_weights = run_nmf_single(X, nmf)
    latent_df      = pd.DataFrame(latent_weights, columns=(
                                ['Latent Feature %s' % (i+1) for i in range(4)]))
    
    concat_df = pd.concat([df, latent_df], axis=1)
    return concat_df


def featurize_single_listing(single_df, kd, search_df, nhood_medians, vectorizer, nmf):
    #Add the neighborhood 1-bedroom median price to the single listing dataframe
    single_df['price_1bd_med'] = nhood_medians[single_df['neighborhood'][0]]

    #Add the nearest neighbors median
    single_df = get_single_listing_median_neighbors(single_df, kd, search_df)
    single_df = add_latent_features_single(single_df, vectorizer, nmf)
    
    return single_df


def create_testing_df(df):
    #Subset dataframe into only features that will be used in testing
    testing_df = df[['beds', 'baths', 'parking', 'price_1bd_med',
                     'med_neighbor_price', 'Latent Feature 1', 
                     'Latent Feature 2', 'Latent Feature 3',  
                     'Latent Feature 4', 'price']]
    return testing_df


