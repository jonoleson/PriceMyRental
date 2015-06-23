import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KDTree
import cPickle


def random_forest_regressor(df):
    y = df.pop('price').values
    X = df.values
    feature_names = df.columns
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3) #Need to worry about random state?

    clf = RandomForestRegressor()
    clf.fit(xtrain, ytrain)
    cPickle.dump(clf, open('models/rfr_model.pkl', 'wb'))
    score = clf.score(xtest, ytest)
    feat_imps = clf.feature_importances_
    rmse = np.mean((ytest - clf.predict(xtest))**2)**0.5
    return 'R^2 is ', score, 'RMSE is ', rmse,'Feature Importances are ', zip(feature_names, feat_imps)




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
    #cPickle.dump(kd, open('models/kd_tree.pkl', 'wb'))
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



def graph_trend(df):
    dfcopy = df.copy()
    df_grouped_median = dfcopy[(dfcopy['beds']==1)].groupby('year-month').median()
    plt.figure(figsize=(10,10))
    df_grouped_median['price'].plot()
