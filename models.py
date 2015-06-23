import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
import cPickle


def random_forest_regressor(df):
    y = df.pop('price').values
    X = df.values
    feature_names = df.columns
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3) 

    clf = RandomForestRegressor()
    clf.fit(xtrain, ytrain)
    #cPickle.dump(clf, open('models/rfr_model.pkl', 'wb'))
    score = clf.score(xtest, ytest)
    feat_imps = clf.feature_importances_
    rmse = np.mean((ytest - clf.predict(xtest))**2)**0.5
    return 'R^2 is ', score, 'RMSE is ', rmse,'Feature Importances are ', zip(feature_names, feat_imps)


def ridge_regressor(df):
    y = df.pop('price').values
    X = df.values
    feature_names = df.columns
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

    clf = Ridge(alpha=1.0)
    clf.fit(xtrain, ytrain)
    #cPickle.dump(clf, open('models/ridge_model.pkl', 'wb'))
    score = clf.score(xtest, ytest)
    feat_imps = clf.coef_
    rmse = np.mean((ytest - clf.predict(xtest))**2)**0.5
    return 'R^2 is ', score, 'RMSE is ', rmse,'Feature coefficients are ', zip(feature_names, feat_imps)

def graph_trend(df):
    dfcopy = df.copy()
    df_grouped_median = dfcopy[(dfcopy['beds']==1)].groupby('year-month').median()
    plt.figure(figsize=(10,10))
    df_grouped_median['price'].plot()
