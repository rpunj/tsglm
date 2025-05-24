"""Example usage of the Python TSGLM implementation."""
import pandas as pd
import numpy as np
from tsglm_py import tsglm, predict_tsglm, lag_matrix

# Load data
sample = pd.read_csv("./data/sample_data.csv")

ts_cases = sample["Cases"]
ts_pop = sample["Population"]
ts_rainfall = sample["Rainfall"]

# Incidence per 100000 inhabitants
ts_incidence = ts_cases / ts_pop * 1e5

n_train = 48
n_test = 12

y_train = ts_cases.iloc[:n_train]
y_test = ts_incidence.iloc[-n_test:]

# Rainfall lag
xreg_lag = lag_matrix(ts_rainfall, [1], label="xreg").iloc[: len(sample)]
xreg_train = xreg_lag.iloc[:n_train]
xreg_test = xreg_lag.iloc[-n_test:]

# Population offset
pop_train = ts_pop.iloc[:n_train]

model = tsglm(
    y_train,
    lags=[1],
    xreg=xreg_train,
    offset=np.log(pop_train),
)

newxreg = pd.concat([xreg_test.reset_index(drop=True), pd.Series(np.log([1e5] * n_test), name="offset")], axis=1)

forecast = predict_tsglm(model, h=n_test, xreg=newxreg.iloc[:, :-1], offset=newxreg["offset"])

print(pd.DataFrame({"observed": y_test.values, "forecast": forecast}))
