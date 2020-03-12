from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model


def model():
    inputs = Input(shape=(1, X_train.shape[2]))
    lstm = LSTM(4)(inputs)
    outputs = Dense(1, activation='linear')(lstm)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

