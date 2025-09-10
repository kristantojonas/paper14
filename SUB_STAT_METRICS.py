from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
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