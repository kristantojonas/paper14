import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def train_xgboost(X_train, X_test, y_train, y_test, n_estimators=200, learning_rate=0.1, max_depth=6):
    y_train_pred = pd.DataFrame(index=y_train.index)
    y_test_pred = pd.DataFrame(index=y_test.index)
    importances = pd.DataFrame(0, index=X_train.columns, columns=y_train.columns)

    for col in y_train.columns:
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train[col])
        y_train_pred[col] = model.predict(X_train)
        y_test_pred[col] = model.predict(X_test)
        importances[col] = model.feature_importances_

    return y_train_pred, y_test_pred, None, importances, None

def train_lightgbm(X_train, X_test, y_train, y_test, n_estimators=200, learning_rate=0.05, max_depth=-1):
    y_train_pred = pd.DataFrame(index=y_train.index)
    y_test_pred = pd.DataFrame(index=y_test.index)
    importances = pd.DataFrame(0, index=X_train.columns, columns=y_train.columns)

    for col in y_train.columns:
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train[col])
        y_train_pred[col] = model.predict(X_train)
        y_test_pred[col] = model.predict(X_test)
        importances[col] = model.feature_importances_

    return y_train_pred, y_test_pred, None, importances, None

def train_catboost(X_train, X_test, y_train, y_test, iterations=300, learning_rate=0.05, depth=6):
    y_train_pred = pd.DataFrame(index=y_train.index)
    y_test_pred = pd.DataFrame(index=y_test.index)
    importances = pd.DataFrame(0, index=X_train.columns, columns=y_train.columns)

    for col in y_train.columns:
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=0,
            random_state=42
        )
        model.fit(X_train, y_train[col])
        y_train_pred[col] = model.predict(X_train)
        y_test_pred[col] = model.predict(X_test)
        importances[col] = model.feature_importances_

    return y_train_pred, y_test_pred, None, importances, None

# ========== SUPPORT VECTOR REGRESSION ==========
def train_svr(X_train, X_test, y_train, y_test, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Trains Support Vector Regression (SVR).
    Note: SVR natively supports only single-output regression, so
    if multiple outputs exist, it loops through each column.
    """
    y_train_pred = pd.DataFrame(index=y_train.index)
    y_test_pred = pd.DataFrame(index=y_test.index)

    for col in y_train.columns:
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train, y_train[col])

        y_train_pred[col] = model.predict(X_train)
        y_test_pred[col] = model.predict(X_test)

    # SVR doesn’t provide feature importances
    return y_train_pred, y_test_pred, None, None, None


# ========== GRADIENT BOOSTED TREES ==========
def train_gbt(X_train, X_test, y_train, y_test, n_est=26, max_de=121):
    """
    Trains Gradient Boosted Trees (GBT).
    Handles multi-output regression by looping through each target column.
    """
    y_train_pred = pd.DataFrame(index=y_train.index)
    y_test_pred = pd.DataFrame(index=y_test.index)
    importances = pd.DataFrame(0, index=X_train.columns, columns=y_train.columns)

    for col in y_train.columns:
        model = GradientBoostingRegressor(
            n_estimators=n_est,
            max_depth=max_de,
        )
        model.fit(X_train, y_train[col])

        y_train_pred[col] = model.predict(X_train)
        y_test_pred[col] = model.predict(X_test)

        importances[col] = model.feature_importances_

    return y_train_pred, y_test_pred, None, importances, None

# ========== RANDOM FOREST ==========
def train_random_forest(X_train, X_test, y_train, y_test,n_est=26,max_de=121):
    model = RandomForestRegressor(n_estimators=n_est,
                                  max_depth=max_de,)
    model.fit(X_train, y_train)

    y_train_pred = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y_train.columns)
    y_test_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)

    significance = pd.DataFrame(
        model.feature_importances_,
        index=X_train.columns,
        columns=['importance']
    )
    return y_train_pred, y_test_pred, None, significance, None

# ========== LINEAR REGRESSION ==========
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = pd.DataFrame(model.predict(X_train), index=y_train.index, columns=y_train.columns)
    y_test_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=y_test.columns)

    coefs = model.coef_
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)

    significance = pd.DataFrame(
        coefs.T,
        index=X_train.columns,
        columns=y_train.columns
    )
    return y_train_pred, y_test_pred, None, significance, None

# ========== CUSTOM CALLBACK FOR ANN ==========
class MetricsCallback(Callback):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_test_pred = self.model.predict(self.X_test, verbose=0)

        r2_train = r2_score(self.y_train, y_train_pred)
        r2_test = r2_score(self.y_test, y_test_pred)
        mse_train = mean_squared_error(self.y_train, y_train_pred)
        mse_test = mean_squared_error(self.y_test, y_test_pred)
        # sse_train = np.sum((self.y_train - y_train_pred) ** 2)
        # sse_test = np.sum((self.y_test - y_test_pred) ** 2)

        self.history.append({
            'epoch': epoch + 1,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            # 'sse_train': sse_train,
            # 'sse_test': sse_test
        })

# ========== ANN ==========
def train_ann(X_train, X_test, y_train, y_test, epochs=100):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    metrics_cb = MetricsCallback(X_train_scaled, y_train.values, X_test_scaled, y_test.values)

    model.fit(X_train_scaled, y_train, epochs=epochs, verbose=0, callbacks=[metrics_cb])

    y_train_pred = pd.DataFrame(
        model.predict(X_train_scaled, verbose=0),
        index=y_train.index,
        columns=y_train.columns
    )
    y_test_pred = pd.DataFrame(
        model.predict(X_test_scaled, verbose=0),
        index=y_test.index,
        columns=y_test.columns
    )

    metrics_df = pd.DataFrame(metrics_cb.history)

    return y_train_pred, y_test_pred, None, metrics_df, None

def train_ann_mlp(X_train, X_test, y_train, y_test, epochs=100, hlz=(41,21,7)):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=hlz,
                         activation='relu',
                         solver='adam',
                         max_iter=1,
                         warm_start=True)

    history = []

    for epoch in range(epochs):
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        if y_train_pred.ndim == 1:
            y_train_pred = y_train_pred.reshape(-1, 1)
            y_test_pred = y_test_pred.reshape(-1, 1)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        # sse_train = np.sum((y_train.values - y_train_pred) ** 2)
        # sse_test = np.sum((y_test.values - y_test_pred) ** 2)

        history.append({
            'epoch': epoch + 1,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            # 'sse_train': sse_train,
            # 'sse_test': sse_test
        })

    y_train_pred = pd.DataFrame(y_train_pred, index=y_train.index, columns=y_train.columns)
    y_test_pred = pd.DataFrame(y_test_pred, index=y_test.index, columns=y_test.columns)
    metrics_df = pd.DataFrame(history)

    return y_train_pred, y_test_pred, None, metrics_df, None
