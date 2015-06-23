import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktLanguageVars
import re
from sklearn.decomposition import NMF
from sklearn.neighbors import KDTree 
import cPickle

def median_neighbors(df, n_neighbors):
    '''
    This model forecasts price by searching for n_neighbors closest 
    listings to each listing and returning the median price of those 
    listings. It does this by using a KD-Tree to efficiently find the
    listings closest to the current listing in location (lat, long)
    and size (beds, baths). It returns an RMSE to test its baseline 
    performance as a standalone model and returns a modified DataFrame
    that can be passed in to another model.
    '''
    kd_df = df[['baths', 'beds', 'lat', 'long']]
    kdvals = kd_df.values
    kd = KDTree(kdvals, leaf_size = 30000)
    cPickle.dump(kd, open('models/kd_tree.pkl', 'wb'))
    neighbors = kd.query(kdvals, k=n_neighbors)

    median_neighbor_prices = []
    for i in xrange(len(df)):
        med_price = df.price[neighbors[1][i][:n_neighbors]].median()
        median_neighbor_prices.append(med_price)

    df['med_neighbor_price'] = median_neighbor_prices
       
    rmse = np.mean((df['med_neighbor_price'] - df['price'])**2)**0.5
    print 'RMSE is ', rmse
    return df

def alt_median_neighbors(df, n_neighbors):
    '''
    This model forecasts price by searching for n_neighbors closest 
    listings to each listing and returning the median price of those 
    listings. It does this by using a KD-Tree to efficiently find the
    listings closest to the current listing by location (lat, long). 
    The indices of these closest listings are then used to find the 
    5 closest listings that have the same layout (same # of beds and baths).
    It returns an RMSE to test its baseline performance as a standalone 
    model and returns a modified DataFrame that can be passed in to 
    another model.
    '''
    kd_df = df[['lat', 'long']]
    kdvals = kd_df.values
    kd = KDTree(kdvals, leaf_size = 1000)
    cPickle.dump(kd, open('models/kd_tree.pkl', 'wb'))
    neighbors = kd.query(kdvals, k=n_neighbors)

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
    X = vectorizer.fit_transform(clean_docs)
    return vectorizer, X

def run_nmf(X, vectorizer, n_topics=4, print_top_words=False):
    nmf = NMF(n_components=n_topics)
    nmf.fit(X)
    cPickle.dump(nmf, open('models/nmf.pkl', 'wb'))
    H = nmf.transform(X)

    if print_top_words==True:
        feature_names = vectorizer.get_feature_names()
        n_top_words = 10    
        for topic_idx, topic in enumerate(nmf.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()

    return H #Return the H matrix

def add_latent_features(df, n_topics):
    clean_text = clean(df.body.values)
    vectorizer, X = get_tfidf(clean_text)
    latent_weights = run_nmf(X, vectorizer, n_topics=n_topics)
    latent_df = pd.DataFrame(latent_weights, columns=(['Latent Feature %s' % (i+1) for i in range(len(n_topics))]))
    df = df.reset_index()
    del df['neighborhood']
    concat_df = pd.concat([df, latent_df], axis=1)
    return concat_df


def featurize_df(df, n_topics, n_neighbors):
    df = median_neighbors(df, n_neighbors=21)
    pass
