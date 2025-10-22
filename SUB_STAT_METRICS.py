from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def calculate_metrics_old(y_train, y_train_pred, y_test, y_test_pred):
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # sse_train = np.sum((y_train - y_train_pred) ** 2)[0]
    # sse_test = np.sum((y_test - y_test_pred) ** 2)[0]

    metrics = pd.DataFrame({
        'MSE': [mse_train, mse_test],
        'R2': [r2_train, r2_test],
        # 'SSE': [sse_train, sse_test]
    }, index=['train', 'test'])

    return metrics

def _safe_mape(y_true, y_pred):
    """Return MAPE as a fraction (not percent). If y_true has zeros,
       those terms become NaN and are ignored in the mean."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / denom))

def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
    # TRAIN
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = _safe_mape(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)

    # TEST
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = _safe_mape(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)

    metrics = pd.DataFrame({
        'MAE':  [mae_train, mae_test],
        'MAPE': [mape_train, mape_test],
        'RMSE': [rmse_train, rmse_test],
        'R2':   [r2_train, r2_test]
    }, index=['train', 'test'])

    return metrics

