from data_management import load_datasets
import os
from model import lstm_model
import ETL as pp
import config
import pandas as pd
pd.pandas.set_option('display.max_columns', None)

"""train pipeline on data"""


def run_training() -> None:
    sales = load_datasets()

    X = pp.reorder(sales, config.COLUMNS)
    X = pp.sorter(X)
    X = pp.grouper(X, config.GROUPING_VARS)
    X = pp.stationarizer(X)
    X = pp.featureBuilder(X, 12)
    _, X = pp.scale_features(X)
    X_train, y_train = pp.targetDefiner(X)
    print(X_train.shape, y_train.shape)
    model = lstm_model((1, X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=1, epochs=30, verbose=1, shuffle=False)
    model.save(config.MODEL_PATH)


if __name__ == '__main__':
    run_training()
