import os
import pandas as pd
import numpy as np
import logging
import sklearn
from sklearn.preprocessing import MinMaxScaler


def reorder(X: pd.DataFrame, columns: list) -> pd.DataFrame:
    X = X.copy()
    X = X[columns]

    return X


def sorter(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['date'] = pd.to_datetime(X['date'], format='%d.%m.%Y')
    X = X.sort_values(by='date').reset_index(drop=True)
    X['date'] = X['date'].dt.year.astype('str') + '-' + X['date'].dt.month.astype('str') + '-01'
    X['date'] = pd.to_datetime(X['date'])

    return X


def grouper(X: pd.DataFrame, variables: list) -> pd.DataFrame:
    X = X.copy()
    X = pd.DataFrame(X.groupby(variables).agg(
        {'item_price': 'mean', 'item_cnt_day': 'sum'}))
    X = X.reset_index(drop=False).rename(columns={'item_cnt_day': 'monthly_sales'})
    X = X.groupby('date').monthly_sales.sum().reset_index()

    return X


def stationarizer(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['prev_sales'] = X['monthly_sales'].shift(1)
    # drop the null values and calculate the difference
    X = X.dropna()
    X['diff'] = (X['monthly_sales'] - X['prev_sales'])

    return X


# def shopGrouper(X: pd.DataFrame) -> pd.DataFrame:
#     X = X.copy()
#     X = X.groupby(['date', 'date_block_num', 'shop_id']).item_cnt_month.sum().reset_index()
#     agg_sales = X.groupby('date_block_num').item_cnt_month.sum().reset_index()
#     date_agg = dict(zip(agg_sales.date_block_num, agg_sales.item_cnt_month))
#     X['agg'] = X.date_block_num.map(date_agg)
#     X['shop_contrib'] = X['item_cnt_month'] / X['agg']
#
#     return X


def featureBuilder(X: pd.DataFrame, look_back: int) -> pd.DataFrame:
    X = X.copy()
    X = X.drop(['prev_sales'], axis=1)
    for i in range(1, look_back + 1):
        fieldname = 'lag_' + str(i)
        X[fieldname] = X['diff'].shift(i)
    # drop Nan
    X = X.dropna(axis=0, inplace=False).reset_index(drop=True)
    X = X.drop(['monthly_sales', 'date'], axis=1)

    return X


def scale_features(X: pd.DataFrame) -> (sklearn.preprocessing.MinMaxScaler, np.array):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X.values)
    train_set_scaled = scaler.transform(X.values)

    return scaler, train_set_scaled


def targetDefiner(X: np.array)-> (np.array, np.array):
    X = X.copy()
    X_train, y_train = X[:, 1:], X[:, 0:1]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

    return X_train, y_train
