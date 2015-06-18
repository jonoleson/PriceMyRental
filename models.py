import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktLanguageVars
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

def clean(list_of_docs):
    clean_docs = []
    for doc in list_of_docs:
        clean_data = re.sub("[^a-zA-Z]"," ", doc) #grabs only letters
        clean_data = re.sub(' +',' ', clean_data) #removes multiple spaces
        clean_data = clean_data.lower() #converts to lower case
         clean_docs.append(clean_data)
    return clean_docs


def tokenize(doc):
    plv = PunktLanguageVars
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in plv.word_tokenize(doc.lower())]


def get_tfidf(data):
    vectorizer = TfidfVectorizer('corpus', tokenizer = tokenize,
                                  stop_words=stopwords.words('english'), 
                                  strip_accents='unicode', norm=l2)
    tfidf = vectorizer.fit_transform(data)
    return tfidf.toarray()

def rf_predict(df):
    y = df.pop('price')
    X = df.values()
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3) #Need to worry about random state?

    clf = RandomForestRegressor()
    clf.fit(xtrain, ytrain)

    return clf.score(xtest, ytest), clf.feature_importances_

