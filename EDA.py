import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import os
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

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


def group_into_monthly_sales(df: pd.DataFrame, *args: list) -> pd.DataFrame:
    X = df.copy()
    grouped = pd.DataFrame(X.groupby(args).agg(
        {'item_price': 'mean', 'item_cnt_day': 'sum'}))
    monthly = grouped.reset_index(drop=False).rename(columns={'item_cnt_day': 'monthly_sales'})

    return monthly


# sales_m = group_into_monthly_sales(sales, ['date', 'date_block_num', 'shop_id', 'item_id'])

def plot_series(df: pd.DataFrame):
    # Group also the aggregate of all shops and items to plot.
    X = df.copy()
    X = X.groupby('date').monthly_sales.sum().reset_index()

    # plot monthly sales
    plot_data = [
        go.Scatter(
            x=X['date'],
            y=X['monthly_sales'],
        )
    ]
    plot_layout = go.Layout(
        title='Monthly Sales'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)


def render_stationary(df: pd.DataFrame) -> pd.DataFrame:
    # create a new dataframe to model the difference
    df_diff = df.copy()
    # add previous sales to the next row
    df_diff['prev_sales'] = df_diff['monthly_sales'].shift(1)
    # drop the null values and calculate the difference
    df_diff = df_diff.dropna()
    df_diff['diff'] = (df_diff['monthly_sales'] - df_diff['prev_sales'])

    # plot sales diff
    plot_data = [
        go.Scatter(
            x=df_diff['date'],
            y=df_diff['diff'],
        )
    ]
    plot_layout = go.Layout(
        title='Montly Sales Diff'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)

    return df_diff
