"""Compare forecasts from R and Python implementations."""
import numpy as np
import pandas as pd
import subprocess
import sys

from tsglm_py import tsglm, predict_tsglm, lag_matrix


def python_forecast():
    sample = pd.read_csv("data/sample_data.csv")
    cases = sample["Cases"]
    pop = sample["Population"]
    rain = sample["Rainfall"]

    n_train = 48
    n_test = 12

    y_train = cases.iloc[:n_train]
    xreg_lag = lag_matrix(rain, [1], label="xreg").iloc[: len(sample)]
    xreg_train = xreg_lag.iloc[:n_train]
    xreg_test = xreg_lag.iloc[-n_test:]
    pop_train = pop.iloc[:n_train]

    model = tsglm(y_train, lags=[1], xreg=xreg_train, offset=np.log(pop_train))
    forecast = predict_tsglm(
        model,
        h=n_test,
        xreg=xreg_test.reset_index(drop=True),
        offset=[np.log(1e5)] * n_test,
    )
    return forecast


def r_forecast():
    try:
        out = subprocess.check_output(["Rscript", "R/generate_forecast.R"], text=True)
    except FileNotFoundError:
        sys.stderr.write("Rscript not found. Please install R to run this check.\n")
        raise
    return np.fromstring(out.strip(), sep=",")


def main():
    py_fct = python_forecast()
    try:
        r_fct = r_forecast()
    except Exception as e:
        sys.stderr.write(f"Failed to run R forecast: {e}\n")
        sys.exit(1)

    equal = np.allclose(py_fct, r_fct, atol=1e-6)
    if equal:
        print("Python and R forecasts match")
    else:
        print("Forecasts differ")
        print("Python:", py_fct)
        print("R:", r_fct)
        sys.exit(1)


if __name__ == "__main__":
    main()
