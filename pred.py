from transform import scale_features, build_featureset
from EDA import render_stationary
from model import model
from data_management import load_datasets
import ETL as pp
import numpy as np
from tensorflow.keras.models import load_model
import config


def load_saved_model():
    model = load_model(config.MODEL_PATH)

    return model


def build_inference_sample() -> np.array:
    sales = load_datasets()

    X = pp.reorder(sales, config.COLUMNS)
    X = pp.sorter(X)
    X = pp.grouper(X, config.GROUPING_VARS)
    X = pp.stationarizer(X)
    X = pp.featureBuilder(X, 12)
    _, X = pp.scale_features(X)
    # the inference sample features are the last records target + first eleven features
    infer_sample = X[-1][:-1]

    return infer_sample
