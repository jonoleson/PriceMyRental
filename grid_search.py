import cPickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor as RFR

def grid_search(X, y):
    '''
    cross validated grid search using Ridge Regressor and Random
    Forest Regressor
    '''

    pars = {'alpha': [0.8, 0.6, 0.5, 0.45, 0.4, 0.2, 0.1,
                      0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]}

    gs = GridSearchCV(Ridge(), pars, cv=5)
    gs.fit(X, y)

    ridge = gs.best_estimator_
    print gs.grid_scores_
    print gs.best_params_
    print gs.best_score_

    cPickle.dump(ridge, open('models/ridge.pkl', 'wb'))

    pars = {'max_depth': [5, 8, 10, 20, 50, 100],
            'min_samples_split': [2, 3, 5, 10, 20]}

    gs = GridSearchCV(RFR(n_estimators=100, random_state=42, n_jobs=2),
                      pars, cv=5)
    gs.fit(X, y)
    rfr = gs.best_estimator_
    print gs.grid_scores_
    print gs.best_params_
    print gs.best_score_

    cPickle.dump(rfr, open('models/rfr.pkl', 'wb'))
    return ridge, rfr


if __name__ == '__main__':
    df = pd.read_csv('data/complete_df.csv', header=False)
    testing_df = df[['beds', 'baths', 'parking', 'washer_dryer', 
                    'price_1bd_med', 'Latent Feature 1', 'Latent Feature 2',
                    'Latent Feature 3', 'Latent Feature 4', 
                    'med_neighbor_price', 'price']]
    testing_df = testing_df.dropna(axis=0)
    y = testing_df.pop('price').values
    X = testing_df.values
    ridge, rfr = grid_search(X, y)