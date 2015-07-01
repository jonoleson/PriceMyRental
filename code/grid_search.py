import cPickle
import pandas as pd
import numpy as np
from featurize import create_testing_df
from sklearn.linear_model import Ridge
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor as RFR


def grid_search(X, y):
    '''
    INPUT: Array of feature values, vector of targets
    OUTPUT: Returns saved pickled random forest at ridge regressor 
    models with the optimal parameters, as well as the models themselves
    '''

    n_samples = X.shape[0]
    cv = ShuffleSplit(n_samples, n_iter=4, test_size=0.3, random_state=0)

    pars = {'alpha': [0.8, 0.6, 0.5, 0.45, 0.4, 0.2, 0.1,
                      0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]}

    gs = GridSearchCV(Ridge(), pars, cv=cv)
    gs.fit(X, y)

    ridge = gs.best_estimator_
    print gs.best_params_
    print gs.best_score_

    cPickle.dump(ridge, open('models/ridge.pkl', 'wb'))

    pars = {'max_depth': [5, 8, 10, 20, 50, 100],
            'min_samples_split': [2, 3, 5, 10, 20]}

    gs = GridSearchCV(RFR(n_estimators=10),
                      pars, cv=cv)
    gs.fit(X, y)
    rfr = gs.best_estimator_
    print gs.best_params_
    print gs.best_score_

    cPickle.dump(rfr, open('models/rfr.pkl', 'wb'))
    return ridge, rfr


if __name__ == '__main__':
    df = pd.read_csv('data/complete_df.csv', header=False)
    testing_df = create_testing_df(df)
    y = testing_df.pop('price').values
    X = testing_df.values
    ridge, rfr = grid_search(X, y)
    grid_search(X, y)