import numpy as np
import pytest

from mkt_surveillance_ml.time_series.lstm import (
    sigmoid,
    simple_rnn_step,
    forget_gate,
    input_gate_and_candidate,
    update_cell_state,
    output_gate_and_hidden_state,
    demonstrate_vanishing_gradient_comparison,
    create_sequences,
    create_multivariate_sequences,
    build_lstm_model,
    fit_lstm_forecaster,
)


class TestSigmoid:
    def test_output_bounded_zero_one(self):
        z = np.array([-100, -1, 0, 1, 100])
        result = sigmoid(z)
        assert (result >= 0).all() and (result <= 1).all()

    def test_sigmoid_of_zero_is_half(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_no_overflow_on_extreme_values(self):
        result = sigmoid(np.array([-1e10, 1e10]))
        assert np.isfinite(result).all()


class TestSimpleRnnStep:
    def test_output_bounded_by_tanh_range(self):
        result = simple_rnn_step(h_prev=2.0, x_t=3.0, W_h=0.5, W_x=0.3, b=0.1)
        assert -1 <= result <= 1


class TestForgetGate:
    def test_output_bounded_zero_one(self):
        result = forget_gate(h_prev=1.0, x_t=np.array([1.0, -2.0, 3.0]), W_f=0.5, U_f=0.3, b_f=0.0)
        assert (result >= 0).all() and (result <= 1).all()

    def test_large_positive_bias_forgets_almost_nothing(self):
        result = forget_gate(h_prev=0.0, x_t=np.array([0.0]), W_f=0.0, U_f=0.0, b_f=10.0)
        assert result[0] == pytest.approx(1.0, abs=1e-3)

    def test_large_negative_bias_forgets_almost_everything(self):
        result = forget_gate(h_prev=0.0, x_t=np.array([0.0]), W_f=0.0, U_f=0.0, b_f=-10.0)
        assert result[0] == pytest.approx(0.0, abs=1e-3)


class TestInputGateAndCandidate:
    def test_input_gate_bounded_zero_one(self):
        i_t, _ = input_gate_and_candidate(
            h_prev=1.0, x_t=np.array([1.0, 2.0]), W_i=0.5, U_i=0.3, b_i=0.0, W_c=0.4, U_c=0.2, b_c=0.0
        )
        assert (i_t >= 0).all() and (i_t <= 1).all()

    def test_candidate_bounded_negative_one_one(self):
        _, c_candidate = input_gate_and_candidate(
            h_prev=1.0, x_t=np.array([1.0, 2.0]), W_i=0.5, U_i=0.3, b_i=0.0, W_c=0.4, U_c=0.2, b_c=0.0
        )
        assert (c_candidate >= -1).all() and (c_candidate <= 1).all()


class TestUpdateCellState:
    def test_forget_one_input_zero_is_pure_carry_forward(self):
        """f_t=1, i_t=0 should leave c_prev completely unchanged --
        the "keep everything, add nothing new" case."""
        c_prev = np.array([3.5, -2.0])
        f_t = np.array([1.0, 1.0])
        i_t = np.array([0.0, 0.0])
        c_candidate = np.array([99.0, 99.0])  # should be fully ignored
        result = update_cell_state(c_prev, f_t, i_t, c_candidate)
        assert np.allclose(result, c_prev)

    def test_forget_zero_input_one_is_pure_replacement(self):
        """f_t=0, i_t=1 should discard c_prev entirely and adopt
        c_candidate completely."""
        c_prev = np.array([3.5, -2.0])
        f_t = np.array([0.0, 0.0])
        i_t = np.array([1.0, 1.0])
        c_candidate = np.array([1.5, 0.5])
        result = update_cell_state(c_prev, f_t, i_t, c_candidate)
        assert np.allclose(result, c_candidate)


class TestOutputGateAndHiddenState:
    def test_hidden_state_bounded_negative_one_one(self):
        result = output_gate_and_hidden_state(
            c_t=np.array([5.0, -5.0]), h_prev=1.0, x_t=np.array([1.0, 1.0]), W_o=0.5, U_o=0.3, b_o=0.0
        )
        assert (result >= -1).all() and (result <= 1).all()

    def test_zero_cell_state_gives_zero_hidden_state(self):
        """tanh(0) = 0, so regardless of the output gate's value, hidden
        state must be exactly 0 when cell state is 0."""
        result = output_gate_and_hidden_state(
            c_t=np.array([0.0]), h_prev=1.0, x_t=np.array([1.0]), W_o=0.5, U_o=0.3, b_o=0.0
        )
        assert result[0] == pytest.approx(0.0)


class TestDemonstrateVanishingGradientComparison:
    def test_plain_rnn_signal_vanishes_toward_zero_by_50_steps(self):
        """File 30 Section 1's exact claim: repeated multiplication by
        (weight * tanh_derivative), both typically < 1, should shrink
        the signal to something extremely small by 50 steps back."""
        result = demonstrate_vanishing_gradient_comparison(n_steps=50, W_h=0.5, random_state=161)
        assert abs(result["plain_rnn_signal_at_n_steps"]) < 1e-6

    def test_lstm_forget_near_one_signal_persists_substantially(self):
        """File 30 Section 2's exact claim: with forget_gate_value=0.95,
        signal at 50 steps should be 0.95^50 -- substantially larger
        than the plain RNN's near-total vanishing."""
        result = demonstrate_vanishing_gradient_comparison(n_steps=50, forget_gate_value=0.95, random_state=161)
        assert result["lstm_signal_at_n_steps"] == pytest.approx(0.95 ** 50, rel=1e-6)

    def test_lstm_signal_is_many_orders_of_magnitude_larger_than_plain_rnn(self):
        """The direct, concrete comparison file 30 builds its entire
        argument for the LSTM architecture on."""
        result = demonstrate_vanishing_gradient_comparison(n_steps=50, random_state=161)
        ratio = abs(result["lstm_signal_at_n_steps"]) / max(abs(result["plain_rnn_signal_at_n_steps"]), 1e-300)
        assert ratio > 1000

    def test_trace_arrays_have_n_steps_plus_one_entries(self):
        result = demonstrate_vanishing_gradient_comparison(n_steps=30)
        assert len(result["plain_rnn_trace"]) == 31
        assert len(result["lstm_forget_near_one_trace"]) == 31


class TestCreateSequences:
    def test_correct_number_of_windows(self):
        data = np.arange(20, dtype=float)
        X, y = create_sequences(data, seq_length=5)
        assert len(X) == 15  # 20 - 5
        assert len(y) == 15

    def test_window_and_target_are_correctly_aligned(self):
        data = np.arange(10, dtype=float)
        X, y = create_sequences(data, seq_length=3)
        assert np.array_equal(X[0], [0, 1, 2])
        assert y[0] == 3
        assert np.array_equal(X[-1], [6, 7, 8])
        assert y[-1] == 9


class TestCreateMultivariateSequences:
    def test_correct_shape(self):
        data = np.random.RandomState(1).normal(0, 1, (20, 3))
        X, y = create_multivariate_sequences(data, seq_length=5, target_col_idx=0)
        assert X.shape == (15, 5, 3)
        assert y.shape == (15,)

    def test_target_column_selected_correctly(self):
        data = np.column_stack([np.arange(10, dtype=float), np.arange(100, 110, dtype=float)])
        X, y = create_multivariate_sequences(data, seq_length=3, target_col_idx=1)
        assert y[0] == 103  # data[3, 1]
        assert np.array_equal(X[0], data[0:3])


class TestBuildLstmModel:
    def test_produces_correct_input_output_shape(self):
        model = build_lstm_model(seq_length=10, n_features=1, lstm_units=8)
        assert model.input_shape == (None, 10, 1)
        assert model.output_shape == (None, 1)

    def test_multivariate_input_shape(self):
        model = build_lstm_model(seq_length=10, n_features=3, lstm_units=8)
        assert model.input_shape == (None, 10, 3)

    def test_model_is_compiled(self):
        model = build_lstm_model(seq_length=10, n_features=1)
        assert model.optimizer is not None
        assert model.loss is not None


class TestFitLstmForecaster:
    def test_produces_finite_forecast_and_metrics(self):
        rng = np.random.RandomState(1)
        n = 200
        series = 100 + np.cumsum(rng.normal(0.02, 1, n))
        train, test = series[:160], series[160:]

        result = fit_lstm_forecaster(
            train, test, seq_length=10, lstm_units=4, epochs=2, batch_size=16, verbose=0
        )
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.mae)
        assert len(result.predictions_actual_scale) == len(result.actual_values_actual_scale)

    def test_predictions_are_on_original_price_scale_not_zero_to_one(self):
        """The whole point of inverse_transform: predictions should come
        back on the ORIGINAL scale (roughly matching the price range),
        not stuck in the [0,1] range the model internally trains on."""
        rng = np.random.RandomState(1)
        n = 200
        series = 1000 + np.cumsum(rng.normal(0.5, 5, n))  # prices in the hundreds/thousands range
        train, test = series[:160], series[160:]

        result = fit_lstm_forecaster(
            train, test, seq_length=10, lstm_units=4, epochs=2, batch_size=16, verbose=0
        )
        # predictions should be somewhere in the same broad range as actual
        # prices (hundreds to low thousands), NOT confined to [0,1]
        assert result.predictions_actual_scale.max() > 10

    def test_scaler_is_fit_only_on_training_data_not_test_data(self):
        """The specific data-leakage regression test motivated by file 30
        Section 3's WRONG-vs-RIGHT demonstration: the scaler's fitted
        range must reflect ONLY train_series, even when test_series has
        values far outside that range. If leakage were reintroduced (e.g.
        by fitting the scaler on train+test combined), the scaler's
        data_max_ would be pulled up to include the test-only extreme
        value -- this test would then fail, catching the regression.
        """
        train = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 10)  # range: 10-50
        test = np.array([1000.0, 2000.0, 3000.0] * 10)  # WAY outside train's range

        result = fit_lstm_forecaster(
            train, test, seq_length=5, lstm_units=4, epochs=1, batch_size=8, verbose=0
        )
        # the scaler's fitted max must reflect train's range (50), NOT be
        # pulled up by test's much larger values (which would happen if
        # test data leaked into the .fit() call)
        assert result.scaler.data_max_[0] == pytest.approx(50.0)
        assert result.scaler.data_min_[0] == pytest.approx(10.0)

    def test_raises_when_train_series_too_short_for_seq_length(self):
        train = np.array([1.0, 2.0, 3.0])
        test = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="seq_length"):
            fit_lstm_forecaster(train, test, seq_length=10, epochs=1)

    def test_train_history_contains_loss(self):
        rng = np.random.RandomState(1)
        n = 150
        series = 100 + np.cumsum(rng.normal(0.02, 1, n))
        train, test = series[:120], series[120:]
        result = fit_lstm_forecaster(
            train, test, seq_length=8, lstm_units=4, epochs=2, batch_size=16, verbose=0
        )
        assert "loss" in result.train_history
        assert len(result.train_history["loss"]) == 2  # one entry per epoch
