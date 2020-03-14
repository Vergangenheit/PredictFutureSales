from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def lstm_model(shape: tuple):
    inputs = Input(shape=shape)
    lstm = LSTM(4)(inputs)
    outputs = Dense(1, activation='linear')(lstm)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


# lstm_regressor = KerasClassifier(
#     build_fn=model,
#     batch_size=1, epochs=100, verbose=1, shuffle=False
# )
