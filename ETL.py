import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import config


class Reorder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.columns]

        return X


class Sorter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X['date'] = pd.to_datetime(X['date'], format='%d.%m.%Y')
        X = X.sort_values(by='date').reset_index(drop=True)
        X['date'] = X['date'].dt.year.astype('str') + '-' + X['date'].dt.month.astype('str') + '-01'
        X['date'] = pd.to_datetime(X['date'])

        return X


class Grouper(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X = pd.DataFrame(X.groupby(self.variables).agg(
            {'item_price': 'mean', 'item_cnt_day': 'sum'}))
        X = X.reset_index(drop=False).rename(columns={'item_cnt_day': 'monthly_sales'})

        return X


class Stationarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X['prev_sales'] = X['monthly_sales'].shift(1)
        # drop the null values and calculate the difference
        X = X.dropna()
        X['diff'] = (X['monthly_sales'] - X['prev_sales'])

        return X


class ShopGrouper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.groupby(['date', 'date_block_num', 'shop_id']).item_cnt_month.sum().reset_index()
        agg_sales = X.groupby('date_block_num').item_cnt_month.sum().reset_index()
        date_agg = dict(zip(agg_sales.date_block_num, agg_sales.item_cnt_month))
        X['agg'] = X.date_block_num.map(date_agg)
        X['shop_contrib'] = X['item_cnt_month'] / X['agg']

        return X


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, look_back):
        self.look_back = look_back

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(['prev_sales'], axis=1)
        for i in range(1, self.look_back + 1):
            fieldname = 'lag_' + str(i)
            X[fieldname] = X['diff'].shift(i)
        # drop Nan
        X = X.dropna(axis=0, inplace=False).reset_index(drop=True)
        X = X.drop(['sales', 'date'], axis=1)

        return X


class TargetDefiner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X_train, y_train = X[:, 1:], X[:, 0:1]

        return X_train, y_train
