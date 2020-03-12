from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model


def model(shape: tuple):
    inputs = Input(shape=shape)
    lstm = LSTM(4)(inputs)
    outputs = Dense(1, activation='linear')(lstm)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
