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

def median_neighbors(df, n_neighbors):
    '''
    This model forecasts price by searching for n_neighbors closest 
    listings to each listing and returning the median price of those 
    listings
    '''
    print 'running...'
    median_prices = []
    for i in xrange(len(df)): 
        point_id = int(df.ix[i,'id'])
        a_lat = df.ix[i,'lat']
        a_long = df.ix[i,'long']
        n_beds = int(df.ix[i,'beds'])
        n_baths = int(df.ix[i,'baths'])
        sub_df = df[(df['beds']==n_beds)&(df['baths']==n_baths)]
        sub_df = sub_df.reset_index()
        sub_df['dists'] = np.nan
        idx = sub_df[sub_df['id'] == point_id].index.tolist()[0]
        for e in xrange(idx):
            b_lat = sub_df.ix[e,'lat']
            b_long = sub_df.ix[e,'long']
            dist = vincenty((a_lat, a_long), (b_lat, b_long)).meters
            sub_df.ix[e,'dists'] = dist
        for e in xrange(idx+1, len(sub_df)):
            b_lat = sub_df.ix[e,'lat']
            b_long = sub_df.ix[e,'long']
            dist = vincenty((a_lat, a_long), (b_lat, b_long)).meters
            sub_df.ix[e,'dists'] = dist
        sub_df = sub_df.sort('dists')
        med_price = sub_df['price'][:n_neighbors].median()
        median_prices.append(med_price)
        if i % 7500 == 0:
            n = i/7500
            print '%s0%' %n
    df['med_neighbor_price'] = median_prices
    rmse = np.mean((df['med_neighbor_price'] - df['price'])**2)**0.5
    return 'RMSE is ', rmse

def graph_trend(df):
    df_copy = df.copy()
    df_grouped_median = df_copy[(df_copy['beds']==1)].groupby('year-month').median()
    plt.figure(figsize=(10,10))
    df_grouped_median['price'].plot()
