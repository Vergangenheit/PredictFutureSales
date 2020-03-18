from data_management import load_datasets
import ETL as pp
import numpy as np
from tensorflow.keras.models import load_model
import config
import sklearn


def load_saved_model():
    model = load_model(config.MODEL_PATH)

    return model


def build_inference_sample():
    sales = load_datasets()

    X = pp.reorder(sales, config.COLUMNS)
    X = pp.sorter(X)
    X = pp.grouper(X, config.GROUPING_VARS)
    X = pp.stationarizer(X)
    X = pp.featureBuilder(X, 12)
    scaler, X = pp.scale_features(X)
    # the inference sample features are the last records target + first eleven features
    infer_sample = X[-1][:-1]

    return scaler, X


def calculate_pred(train_set_scaled: np.array, scaler: sklearn.preprocessing.MinMaxScaler):
    infer_sample = train_set_scaled[-1][:-1]
    model = load_model(config.MODEL_PATH)
    features_2015_11 = infer_sample.reshape(1, 1, infer_sample.shape[0]).astype('float32')
    # calculate prediction
    y_pred = model.predict(features_2015_11, batch_size=1)
    features_2015_11 = features_2015_11.reshape(features_2015_11.shape[1], features_2015_11.shape[2])
    pred_test_set = np.hstack((y_pred, features_2015_11))
    # stack two precedent samples to the pred test to see if inverse scaler gives back the expected data
    pred_test_set = np.vstack((train_set_scaled[-10:], pred_test_set))
    # inverse scale the set
    final_pred = scaler.inverse_transform(pred_test_set)

    print(final_pred[-1])


if __name__ == '__main__':
    scaler, train_set_scaled = build_inference_sample()
    calculate_pred(train_set_scaled, scaler)
