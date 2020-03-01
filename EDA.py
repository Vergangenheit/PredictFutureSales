import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import os

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

sales = pd.read_csv(os.path.join('data/', 'sales_train.csv'))
items = pd.read_csv(os.path.join('data/', 'items.csv'))
test = pd.read_csv(os.path.join('data/', 'test.csv'))
item_categories = pd.read_csv(os.path.join('data/', 'item_categories.csv'))
shops = pd.read_csv(os.path.join('data/', 'shops.csv'))

sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')


def merging(df1: pd.DataFrame, df2: pd.DataFrame, key: str) -> pd.DataFrame:
    X = df1.copy()
    X = X.merge(df2, how='left', on=key)
    return X


sales = merging(sales, items, 'item_id')
sales = merging(sales, item_categories, 'item_category_id')
sales = merging(sales, shops, 'shop_id')


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['date',
            'date_block_num',
            'shop_id',
            'shop_name',
            'item_category_id',
            'item_category_name',
            'item_id',
            'item_name',
            'item_price',
            'item_cnt_day']

    df = df[cols]

    return df


def convert_dates_to_months(df: pd.DataFrame) -> pd.DataFrame:
    # represent month in date field as its first day
    X = df.copy()
    X['date'] = pd.to_datetime(X['date'], format='%d.%m.%Y')
    X = X.sort_values(by='date').reset_index(drop=True)
    X['date'] = X['date'].dt.year.astype('str') + '-' + X['date'].dt.month.astype('str') + '-01'
    X['date'] = pd.to_datetime(X['date'])

    return X


def group_into_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    grouped = pd.DataFrame(X.groupby(['date', 'date_block_num', 'shop_id', 'item_id']).agg(
        {'item_price': 'mean', 'item_cnt_day': 'sum'}))
    monthly = grouped.reset_index(drop=False).rename(columns={'item_cnt_day': 'item_cnt_month'})

    return monthly
