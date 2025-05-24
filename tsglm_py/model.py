import numpy as np
import pandas as pd
import statsmodels.api as sm


def lag_matrix(y, lags, label="y"):
    """Create a matrix of lagged values for a series."""
    y = pd.Series(y)
    df = pd.DataFrame({label: y})
    for lag in lags:
        df[f"{label}_{lag}"] = y.shift(lag)
    return df


def tsglm(y, lags, xreg=None, offset=None):
    """Fit a Negative Binomial GLM for time-series forecasting.

    Parameters
    ----------
    y : array-like
        Time series values.
    lags : sequence of int
        Lags to include as regressors.
    xreg : array-like or DataFrame, optional
        Optional external regressors.
    offset : array-like, optional
        Optional offset values.

    Returns
    -------
    statsmodels.GLMResults
        Fitted model with additional attributes ``y``, ``lags`` and
        ``feature_names`` for forecasting.
    """
    y_series = pd.Series(y).reset_index(drop=True)
    df_lag = lag_matrix(y_series, lags)
    df_lag = df_lag.iloc[: len(y_series)]

    exog_parts = []
    if len(lags) > 0:
        lag_names = [f"y_{lag}" for lag in lags]
        exog_parts.append(df_lag[lag_names])
    else:
        lag_names = []

    if xreg is not None:
        xreg_df = pd.DataFrame(xreg)
        if xreg_df.columns.isnull().any():
            xreg_df.columns = [f"x{i}" for i in range(xreg_df.shape[1])]
        exog_parts.append(xreg_df.reset_index(drop=True))
        xreg_names = list(xreg_df.columns)
    else:
        xreg_names = []

    exog = pd.concat(exog_parts, axis=1) if exog_parts else pd.DataFrame(index=df_lag.index)

    offset_series = pd.Series(offset).reset_index(drop=True) if offset is not None else None

    # Drop rows with missing values
    data = pd.concat([y_series, exog], axis=1)
    if offset_series is not None:
        data = pd.concat([data, offset_series.rename("offset")], axis=1)
    data = data.dropna()

    y_train = data.iloc[:, 0]
    exog_train = data.iloc[:, 1:1 + exog.shape[1]] if exog_parts else pd.DataFrame(index=data.index)
    if exog_train.empty:
        exog_train = pd.DataFrame({"const": np.ones(len(y_train))})
    else:
        exog_train = sm.add_constant(exog_train, has_constant="add")

    if offset_series is not None:
        offset_train = data["offset"]
    else:
        offset_train = None

    model = sm.GLM(y_train, exog_train, family=sm.families.NegativeBinomial(), offset=offset_train)
    result = model.fit()

    result.y = y_series.values
    result.lags = list(lags)
    result.feature_names = list(exog_train.columns)
    result.has_offset = offset is not None

    return result


def predict_tsglm(model, h, xreg=None, offset=None):
    """Recursive forecast for a fitted TSGLM model."""
    if xreg is not None:
        xreg_df = pd.DataFrame(xreg)
    else:
        xreg_df = None

    if model.has_offset:
        if offset is None:
            raise ValueError("Offset values must be provided for forecasting.")
        offset_series = pd.Series(offset)
    else:
        offset_series = None

    y = model.y
    lags = model.lags
    maxlag = max(lags) if len(lags) > 0 else 0
    flag = list(reversed(y[-maxlag:])) if maxlag > 0 else []

    preds = []
    for i in range(h):
        row_parts = []
        if maxlag > 0:
            lag_values = [flag[l - 1] for l in lags]
            row_parts.append(pd.DataFrame([lag_values], columns=[f"y_{l}" for l in lags]))
        if xreg_df is not None:
            row_parts.append(xreg_df.iloc[[i]].reset_index(drop=True))
        row = pd.concat(row_parts, axis=1) if row_parts else pd.DataFrame(index=[0])
        if row.empty:
            exog = pd.DataFrame({"const": [1.0]})
        else:
            exog = sm.add_constant(row, has_constant="add")
        if offset_series is not None:
            off = offset_series.iloc[i]
            pred = model.predict(exog, offset=[off])[0]
        else:
            pred = model.predict(exog)[0]
        preds.append(pred)
        if maxlag > 0:
            flag = [pred] + flag[:-1]
    return np.array(preds)
