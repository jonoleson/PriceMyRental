import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktLanguageVars
import re
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from geopy.distance import vincenty

#class PriceMyRental()


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

def run_nmf(X, vectorizer, n_topics=4):
    nmf = NMF(n_components=n_topics)
    H = nmf.fit_transform(X)   

    feature_names = vectorizer.get_feature_names()
    n_top_words = 10    

    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
    return H #Return the H matrix


def random_forest_regressor(df):
    y = df.pop('price').values
    X = df.values
    feature_names = df.columns
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3) #Need to worry about random state?

    clf = RandomForestRegressor()
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    feat_imps = clf.feature_importances_
    rmse = np.mean((ytest - clf.predict(xtest))**2)**0.5
    return 'R^2 is ', score, 'RMSE is ', rmse,'Feature Importances are ', zip(feature_names, feat_imps)

def rf_with_nmf(df, n_topics):
    clean_text = clean(df.body.values)
    vectorizer, X = get_tfidf(clean_text)
    latent_weights = run_nmf(X, vectorizer, n_topics=n_topics)
    latent_df = pd.DataFrame(latent_weights, columns=(['Latent Factor %s' % (i+1) for i in range(len(n_topics))]))
    df = df.reset_index()
    del df['neighborhood']
    concat_df = pd.concat([df, latent_df], axis=1)
    print random_forest_regressor(concat_df)

def median_neighbors(df, n_listings, n_neighbors):
    '''
    This model forecasts price by searching for n_neighbors closest 
    listings to each listing and returning the median price of those 
    listings
    '''
    print 'running...'
    dfcopy = df.copy()
    dfcopy['med_neighbor_price'] = np.nan
    for i in xrange(n_listings):
        point_id = int(dfcopy.ix[i,'id'])
        a_lat    = dfcopy.ix[i,'lat']
        a_long   = dfcopy.ix[i,'long']
        n_beds   = int(dfcopy.ix[i,'beds'])
        n_baths  = int(dfcopy.ix[i,'baths'])
        
        # Create subset of dataframe w/same beds & baths,
        # omitting the current listing
        sub_df = dfcopy[
            (dfcopy['beds']  == n_beds)  &
            (dfcopy['baths'] == n_baths) &
            (dfcopy['id']    != point_id)
        ]
        sub_df.reset_index(inplace=True)
        sub_df['dists'] = np.nan
        
        # calc distance b/w each row in the sub-DF
        # and the current listing
        for e in xrange(len(sub_df)):
            b_lat  = sub_df.ix[e, 'lat']
            b_long = sub_df.ix[e, 'long']
            dist   = vincenty((a_lat, a_long), (b_lat, b_long)).meters
            sub_df.ix[e, 'dists'] = dist
        
        sub_df = sub_df.sort('dists')
        med_price = sub_df['price'][:n_neighbors].median()
        dfcopy.ix[i,'med_neighbor_price'] = med_price
        if i % 750 == 0:
            n = i/750
            print '%s percent' %n
    
    return dfcopy    
    # rmse = np.mean((df['med_neighbor_price'] - df['price'])**2)**0.5
    # return 'RMSE is ', rmse

def graph_trend(df):
    dfcopy = df.copy()
    df_grouped_median = dfcopy[(dfcopy['beds']==1)].groupby('year-month').median()
    plt.figure(figsize=(10,10))
    df_grouped_median['price'].plot()
