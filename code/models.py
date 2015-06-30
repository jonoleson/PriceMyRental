import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
import cPickle


def random_forest_regressor(df):
    '''
    INPUT: Pandas dataframe
    OUTPUT: R^2 and Mean Absolute Error performance metrics, feature importances
    '''

    y                            = df.pop('price').values
    X                            = df.values
    feature_names                = df.columns
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, 
                                                    random_state=5) 

    clf = RandomForestRegressor()
    clf.fit(xtrain, ytrain)
    score       = clf.score(xtest, ytest)
    feat_imps   = clf.feature_importances_
    ypredict    = clf.predict(xtest)
    mae         = np.mean(np.absolute(ytest - ypredict))
    mae_percent = np.mean(np.absolute(ytest - ypredict) / ytest)
    return 'R^2 is ', score, 'MAE is ', mae, 'MAE percent is ', mae_percent, \
           'Feature Importances are ', zip(feature_names, feat_imps)

def build_production_rfr(df):
    '''
    INPUT: Pandas dataframe
    OUTPUT: Saved pickled model that's been trained on the full dataset, 
    to be used in the app 
    '''
    y             = df.pop('price').values
    X             = df.values
    feature_names = df.columns 
    clf           = RandomForestRegressor()

    clf.fit(X, y)
    cPickle.dump(clf, open('../models/rfr.pkl', 'wb'))
    score = clf.score(X, y)
    print score  

def ridge_regressor(df):
    '''
    INPUT: Pandas dataframe
    OUTPUT: R^2 and Mean Absolute Error performance metrics, feature coefficients
    '''
    y                            = df.pop('price').values
    X                            = df.values
    feature_names                = df.columns
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, 
                                                    random_state=0)

    clf = Ridge(alpha=1.0)
    clf.fit(xtrain, ytrain)

    score       = clf.score(xtest, ytest)
    feat_imps   = clf.coef_
    ypredict    = clf.predict(xtest)
    mae         = np.mean(np.absolute(ytest - ypredict))
    mae_percent = np.mean(np.absolute(ytest - ypredict) / ytest)
    return 'R^2 is ', score, 'RMSE is ', rmse, 'MAE percent is ', mae_percent, \
           'Feature coefficients are ', zip(feature_names, feat_imps)

def graph_trend(df):
    '''
    INPUT: Pandas dataframe
    OUTPUT: Graph of monthly trend in median rents citywide
    '''
    dfcopy            = df.copy()
    df_grouped_median = dfcopy[(dfcopy['beds'] == 1)].groupby('year-month').median()
    plt.figure(figsize=(10,10))
    df_grouped_median['price'].plot()

