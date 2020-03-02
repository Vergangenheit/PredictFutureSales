from EDA import merging, reorder_columns, convert_dates_to_months, group_into_monthly_sales
import pandas as pd
import numpy as np

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
