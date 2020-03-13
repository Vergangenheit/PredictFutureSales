from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import data_management as dm
import ETL as pp
import config
"""instantiate Pipeline obect"""

sales_pipe = Pipeline(
    ('reorder', pp.Reorder(config.COLUMNS)),
    ('sorter', pp.Sorter()),
    ('grouper', pp.Grouper(config.GROUPING_VARS)),
    ('shop_grouper', pp.ShopGrouper()),
    ('feature_builder', pp.FeatureBuilder(config.LOOK_BACK)),
    ('scaler', MinMaxScaler(feature_range=(-1, 1))),
    ('feature_target_splitter', pp.TargetDefiner())
)