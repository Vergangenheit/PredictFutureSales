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


def merging(df1: pd.DataFrame, df2: pd.DataFrame, key: str) -> pd.DataFrame:
    X = df1.copy()
    X = X.merge(df2, how='left', on=key)
    return X


sales = merging(sales, items, 'item_id')
sales = merging(sales, item_categories, 'item_category_id')
sales = merging(sales, shops, 'shop_id')
