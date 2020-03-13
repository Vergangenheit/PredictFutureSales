DATASET_DIR = 'data/'
SALES_FILE = 'sales_train.csv'
ITEMS_FILE = 'items.csv'
CATEGORIES_FILE = 'item_categories.csv'
SHOPS_FILE = 'shops.csv'

COLUMNS = ['date',
           'date_block_num',
           'shop_id',
           'shop_name',
           'item_category_id',
           'item_category_name',
           'item_id',
           'item_name',
           'item_price',
           'item_cnt_day']

GROUPING_VARS = ['date', 'date_block_num', 'shop_id', 'item_id']

LOOK_BACK = 12
