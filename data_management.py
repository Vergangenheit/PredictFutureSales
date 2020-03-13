import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import config

_logger = logging.getLogger(__name__)


def load_datasets():
    sales = pd.read_csv(os.path.join(config.DATASET_DIR, config.SALES_FILE))
    items = pd.read_csv(os.path.join(config.DATASET_DIR, config.ITEMS_FILE))
    # test = pd.read_csv(os.path.join(config.DATASET_DIR, 'test.csv'))
    item_categories = pd.read_csv(os.path.join(config.DATASET_DIR, config.CATEGORIES_FILE))
    shops = pd.read_csv(os.path.join(config.DATASET_DIR, config.SHOPS_FILE))

    sales = merging(sales, items, 'item_id')
    sales = merging(sales, item_categories, 'item_category_id')
    sales = merging(sales, shops, 'shop_id')
    sales = reorder_columns(sales)

    return sales


def merging(df1: pd.DataFrame, df2: pd.DataFrame, key: str) -> pd.DataFrame:
    X = df1.copy()
    X = X.merge(df2, how='left', on=key)
    return X
