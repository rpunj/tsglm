import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from tsglm_py import lag_matrix, tsglm, predict_tsglm


def test_lag_matrix_basic():
    y = [1, 2, 3, 4]
    df = lag_matrix(y, [1, 2])
    expected = pd.DataFrame({
        "y": [1, 2, 3, 4],
        "y_1": [np.nan, 1, 2, 3],
        "y_2": [np.nan, np.nan, 1, 2],
    })
    pd.testing.assert_frame_equal(df, expected)


def test_tsglm_forecasting_sample():
    sample = pd.read_csv("data/sample_data.csv")
    n_train, n_test = 48, 12
    cases = sample["Cases"]
    pop = sample["Population"]
    rain = sample["Rainfall"]

    train_cases = cases.iloc[:n_train]
    train_pop = pop.iloc[:n_train]
    train_xreg = lag_matrix(rain, [1], label="xreg").iloc[:n_train]
    forecast_xreg = lag_matrix(rain, [1], label="xreg").iloc[n_train:n_train + n_test]
    test_pop = pop.iloc[n_train:n_train + n_test]

    model = tsglm(train_cases, lags=[1], xreg=train_xreg, offset=np.log(train_pop))
    forecast = predict_tsglm(model, h=n_test, xreg=forecast_xreg, offset=np.log(test_pop))

    expected = np.array([
        7.3310, 5.8751, 7.2924, 10.0071, 7.9333, 4.9603,
        3.9474, 3.0004, 2.8780, 2.8667, 2.9942, 4.0689,
    ])
    np.testing.assert_allclose(forecast, expected, rtol=1e-4, atol=1e-4)
