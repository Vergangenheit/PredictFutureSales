from transform import scale_features, build_featureset
from EDA import render_stationary
from model import model
import numpy as np

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
model = model((1, X_train.shape[2]))
model.fit(X_train, y_train, batch_size=1, epochs=100, verbose=1, shuffle=False)


def build_inference_sample() -> np.array:
    ## TO COMPLETE PIPELINE
    df_diff = render_stationary()
    df_supervised = build_inference_sample(df_diff)
    scaler, train_set_scaled = scale_features(df_supervised)
    # build and train model
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
    model = model((1, X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=1, epochs=100, verbose=1, shuffle=False)
    features_2015_11 = train_set_scaled[-1][:-1]
    features_2015_11 = features_2015_11.reshape(1, 1, features_2015_11.shape[0]).astype('float32')
    # calculate prediction
    y_pred = model.predict(features_2015_11, batch_size=1)
    features_2015_11 = features_2015_11.reshape(features_2015_11.shape[1], features_2015_11.shape[2])
    pred_test_set = np.hstack((y_pred, features_2015_11))
    # inverse scale the sample
    final_pred = scaler.inverse_transform(pred_test_set)

    return final_pred
