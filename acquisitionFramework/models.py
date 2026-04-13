from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

def createRNN(output_dim=15, input_shape=None):
    model = keras.Sequential(name="Fast_GRU_Model")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Dense(128, activation="tanh"))

    model.add(layers.GRU(128, return_sequences=True, dropout=0.1, activation="tanh"))
    model.add(layers.GRU(64, return_sequences=False, dropout=0.1, activation="tanh"))

    model.add(layers.Dense(64, activation="tanh"))
    model.add(layers.Dense(output_dim, activation="linear", dtype="float32"))  # ensure float32 output

    optimizer = Adam(learning_rate=2e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model