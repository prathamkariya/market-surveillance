"""
LSTM Time Series, matching file 30.

The individual gate functions (sigmoid, forget_gate, input_gate_and_candidate,
update_cell_state, output_gate_and_hidden_state) are preserved exactly --
this is the mathematical core worth being able to derive and explain gate
by gate, not something to abstract away behind Keras from the start.

fit_lstm_forecaster is deliberately the ONLY entry point for actually
training an LSTM forecaster in this module. It takes train and test data
as SEPARATE arguments and fits the scaler internally on train_series
alone -- this isn't just documented as the right way (file 30's own
WRONG-vs-RIGHT demonstration shows what happens if you don't), it's
structural: there is no code path in this function that has access to
both train and test data before the scaler is fit, so the leakage bug
the notes specifically warn about cannot silently reappear here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress TensorFlow's informational CPU-optimization logs (oneDNN
# notices, AVX instruction listings) -- accurate, but not actionable by
# a caller of this module, and otherwise clutters every import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def simple_rnn_step(h_prev: float, x_t: float, W_h: float, W_x: float, b: float) -> float:
    """h_t = tanh(W_h @ h_prev + W_x @ x_t + b). File 30 Section 1 -- the
    simplest possible recurrent step, whose vanishing-gradient problem
    (repeated multiplication by weight * tanh-derivative, both typically
    < 1, shrinking a gradient signal toward zero over many steps) is the
    entire motivation for every LSTM gate defined below.
    """
    return np.tanh(W_h * h_prev + W_x * x_t + b)


def forget_gate(h_prev: float, x_t: np.ndarray, W_f: float, U_f: float, b_f: float) -> np.ndarray:
    """f_t = sigmoid(W_f @ x_t + U_f @ h_prev + b_f). File 30 Section 2.1.
    Output in [0,1]: near 0 means "forget this cell-state dimension
    almost entirely," near 1 means "keep it almost entirely."
    """
    return sigmoid(W_f * x_t + U_f * h_prev + b_f)


def input_gate_and_candidate(
    h_prev: float, x_t: np.ndarray, W_i: float, U_i: float, b_i: float,
    W_c: float, U_c: float, b_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """i_t = sigmoid(...) -- HOW MUCH new info to add.
    c_candidate_t = tanh(...) -- WHAT the new info actually is.
    File 30 Section 2.2. tanh's [-1,1] range suits an actual value to
    potentially add (positive or negative); sigmoid's [0,1] range suits
    a gating fraction -- the same functional split as the forget gate.
    """
    i_t = sigmoid(W_i * x_t + U_i * h_prev + b_i)
    c_candidate = np.tanh(W_c * x_t + U_c * h_prev + b_c)
    return i_t, c_candidate


def update_cell_state(c_prev: np.ndarray, f_t: np.ndarray, i_t: np.ndarray, c_candidate: np.ndarray) -> np.ndarray:
    """c_t = f_t * c_prev + i_t * c_candidate. File 30 Section 2.3.

    THE critical structural difference from a plain RNN: this is a
    WEIGHTED SUM (mostly additive), not a repeated multiplication through
    a shared weight matrix every step -- when f_t stays close to 1
    (learned "keep this"), a gradient signal can propagate across many
    steps nearly undiminished, unlike the plain RNN's multiplicative decay.
    """
    return f_t * c_prev + i_t * c_candidate


def output_gate_and_hidden_state(
    c_t: np.ndarray, h_prev: float, x_t: np.ndarray, W_o: float, U_o: float, b_o: float,
) -> np.ndarray:
    """o_t = sigmoid(...); h_t = o_t * tanh(c_t). File 30 Section 2.4.

    c_t is the "long-term memory"; the output gate decides how much of
    it to actually EXPOSE as this step's hidden state (used for the
    current prediction, and passed to the next step's gates) -- an
    internal memory versus an externally-used, filtered view of it.
    """
    o_t = sigmoid(W_o * x_t + U_o * h_prev + b_o)
    return o_t * np.tanh(c_t)


def demonstrate_vanishing_gradient_comparison(
    n_steps: int = 50, W_h: float = 0.5, forget_gate_value: float = 0.95, random_state: int | None = None,
) -> dict:
    """File 30 Sections 1-2's core motivating comparison, as a reusable
    function rather than a one-off script -- this IS the proof for why
    the LSTM architecture (specifically the additive cell-state pathway)
    exists at all, not a bare assertion.

    Plain RNN trace: gradient_signal *= W_h * tanh_derivative at each
    step, where tanh_derivative is evaluated at a representative random
    point each step (1 - tanh(N(0,1))^2, always <= 1, typically well
    below it) -- repeated multiplication by two typically-sub-1
    quantities shrinks the signal geometrically.

    LSTM cell-state trace: c_trace *= forget_gate_value at each step, a
    SINGLE learned scalar close to 1 -- when the model has learned "keep
    this," the gradient pathway through the cell state is repeated
    multiplication by a value NEAR 1, not by a weight-and-nonlinearity
    product that's typically much smaller.
    """
    rng = np.random.RandomState(random_state)

    gradient_signal = 1.0
    rnn_trace = [gradient_signal]
    for _ in range(n_steps):
        tanh_derivative_this_step = 1 - np.tanh(rng.normal(0, 1)) ** 2
        gradient_signal *= W_h * tanh_derivative_this_step
        rnn_trace.append(gradient_signal)

    c_trace = 1.0
    lstm_trace = [c_trace]
    for _ in range(n_steps):
        c_trace *= forget_gate_value
        lstm_trace.append(c_trace)

    return {
        "plain_rnn_trace": np.array(rnn_trace),
        "lstm_forget_near_one_trace": np.array(lstm_trace),
        "plain_rnn_signal_at_n_steps": float(rnn_trace[n_steps]),
        "lstm_signal_at_n_steps": float(lstm_trace[n_steps]),
    }


def create_sequences(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """File 30 Section 4. Windows a 1D array into (X, y) where X[i] is
    seq_length consecutive points and y[i] is the point immediately
    after that window -- the standard supervised-learning framing of a
    sequence forecasting problem.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def create_multivariate_sequences(
    data: np.ndarray, seq_length: int, target_col_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """File 30 Section 7. Same windowing as create_sequences, but over
    multiple feature columns at once -- the LSTM architecture needs no
    structural change to accept this, only input_shape's final dimension
    changes from 1 to n_features, unlike ARIMA which is fundamentally
    built around a single series' own autocorrelation.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length: int, n_features: int = 1, lstm_units: int = 50, dropout_rate: float = 0.2):
    """File 30 Section 5. Stacked LSTM architecture: the first layer
    needs return_sequences=True since the second LSTM layer requires a
    full sequence as ITS input, not a single vector; the second layer's
    return_sequences defaults to False since only a final summary is
    needed for the Dense output layer. Dropout is the neural-network
    analogue of file 23's feature subsampling -- both deliberately
    introduce randomness specifically to prevent over-reliance on a
    narrow subset of the model's own capacity.
    """
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(seq_length, n_features)),
        keras.layers.LSTM(lstm_units, activation="tanh", return_sequences=True),
        keras.layers.Dropout(dropout_rate),
        keras.layers.LSTM(lstm_units, activation="tanh"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@dataclass
class LstmForecastResult:
    fitted_model: object
    scaler: MinMaxScaler
    rmse: float
    mae: float
    predictions_actual_scale: np.ndarray
    actual_values_actual_scale: np.ndarray
    train_history: dict


def fit_lstm_forecaster(
    train_series: np.ndarray, test_series: np.ndarray, seq_length: int = 30,
    lstm_units: int = 50, dropout_rate: float = 0.2, epochs: int = 50,
    batch_size: int = 32, validation_split: float = 0.1, verbose: int = 0,
) -> LstmForecastResult:
    """File 30 Sections 3-6, assembled into the ONE function this module
    exposes for actually fitting an LSTM forecaster -- deliberately
    structured so the scaler-leakage bug file 30 Section 3's WRONG-vs-
    RIGHT demonstration warns about cannot silently reappear: the scaler
    is fit ONLY on train_series (line below), and test_series is only
    ever `.transform()`-ed with that already-fitted scaler, never
    included in a `.fit()` call. There's no code path here with access
    to both arrays before the scaler is fit.

    train_series/test_series: 1D arrays of raw (unscaled) values, e.g.
    df['close'].values[:split] and df['close'].values[split:].
    """
    train_series = np.asarray(train_series).reshape(-1, 1)
    test_series = np.asarray(test_series).reshape(-1, 1)

    if len(train_series) <= seq_length:
        raise ValueError(
            f"train_series has {len(train_series)} points, but seq_length="
            f"{seq_length} requires more than that just to form one training "
            f"sequence. Use a longer training series or a shorter seq_length."
        )

    # RIGHT way (file 30 Section 3): fit the scaler on TRAINING data only.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series).flatten()
    test_scaled = scaler.transform(test_series).flatten()

    X_train_seq, y_train_seq = create_sequences(train_scaled, seq_length)
    # Test sequences include the tail of scaled TRAINING data specifically
    # so a rolling forecast can start right at the test period's beginning --
    # exactly as it would need to at real deployment time.
    combined_for_test = np.concatenate([train_scaled[-seq_length:], test_scaled])
    X_test_seq, y_test_seq = create_sequences(combined_for_test, seq_length)

    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))

    model = build_lstm_model(seq_length, n_features=1, lstm_units=lstm_units, dropout_rate=dropout_rate)
    history = model.fit(
        X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
        validation_split=validation_split, verbose=verbose,
    )

    predictions_scaled = model.predict(X_test_seq, verbose=0)
    predictions_actual = scaler.inverse_transform(predictions_scaled)
    actual_values = scaler.inverse_transform(y_test_seq.reshape(-1, 1))

    rmse = float(np.sqrt(mean_squared_error(actual_values, predictions_actual)))
    mae = float(mean_absolute_error(actual_values, predictions_actual))

    return LstmForecastResult(
        fitted_model=model, scaler=scaler, rmse=rmse, mae=mae,
        predictions_actual_scale=predictions_actual.flatten(),
        actual_values_actual_scale=actual_values.flatten(),
        train_history=history.history,
    )
