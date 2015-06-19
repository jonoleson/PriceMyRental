import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktLanguageVars
import re
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split


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

