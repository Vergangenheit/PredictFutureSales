from EDA import merging, reorder_columns, convert_dates_to_months, group_into_monthly_sales
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

"""this module hosts functions that help to analyze the contributions by shop to the total sales"""


def group_by_shop(sales_m: pd.DataFrame) -> pd.DataFrame:
    sales_m_agg = sales_m.groupby(['date', 'date_block_num', 'shop_id']).item_cnt_month.sum().reset_index()
    agg_sales = sales_m_agg.groupby('date_block_num').item_cnt_month.sum().reset_index()
    date_agg = dict(zip(agg_sales.date_block_num, agg_sales.item_cnt_month))
    sales_m_agg['agg'] = sales_m_agg.date_block_num.map(date_agg)
    sales_m_agg['shop_contrib'] = sales_m_agg['item_cnt_month'] / sales_m_agg['agg']

    return sales_m_agg


def test_contrib(sales_m_agg: pd.DataFrame):
    for num in sales_m_agg.date_block_num.unique():
        assert sales_m_agg[sales_m_agg.date_block_num == 0]['shop_contrib'].sum() == 1.0


def build_featureset(df_diff: pd.DataFrame, look_back: int) -> pd.DataFrame:
    # create dataframe for transformation from time series to supervised
    X = df_diff.copy()
    df_supervised = X.drop(['prev_sales'], axis=1)
    for i in range(1, look_back + 1):
        fieldname = 'lag_' + str(i)
        df_supervised[fieldname] = df_supervised['diff'].shift(i)
    # drop Nan
    df_supervised = df_supervised.dropna(axis=0, inplace=False).reset_index(drop=True)

    return df_supervised


def scale_features(df_supervised: pd.DataFrame) -> (sklearn.preprocessing.MinMaxScaler, np.array):
    df_model = df_supervised.drop(['sales', 'date'], axis=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df_model.values)
    train_set_scaled = scaler.transform(df_model.values)

    return scaler, train_set_scaled


def adjusted_r_sqrt(df_supervised: pd.DataFrame, target: str, features: list):
    streak = features[0]
    for feature in features[1:]:
        streak += ' + ' + feature
    model = smf.ols(formula=target + ' ~ ' + str(streak), data=df_supervised)
    model_fit = model.fit()
    regression_adj_rsq = model_fit.rsquared_adj
    print(regression_adj_rsq)
