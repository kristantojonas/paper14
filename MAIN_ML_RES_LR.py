import pandas as pd
from SUB_ML import train_linear_regression
from SUB_STAT_METRICS import calculate_metrics
from SUB_PREPROCESSING import prepare_data
from SUB_PLOTTING import plot_predictions_vs_actual
from SUB_PREPREPROCESSING import PREPREPROCESSING

# === Load data ===
y_cols = ['Gas', 'Liquid', 'Solid']
X_ops = ['Particle Size (mm)', 'Temperature (C)', 'Residence Time (h)',
         'Carrier Gas (mL/min)', 'Heating Rate (C/h)']
df_raw = pd.read_excel('RAW ML.xlsx', skiprows=1)
df = df_raw[df_raw.columns.difference(['No.', 'First Author'])]
df = PREPREPROCESSING(df)

# === Redevelop model ===
# target = 'Gas'
# best_rs = 34
# target = 'Liquid'
# best_rs = 35
target = 'Solid'
best_rs = 7

X_train, X_test, y_train, y_test = prepare_data(
    df, target, y_cols, best_rs,
    exclude_hier_cols=X_ops, test_size=0.25
)

# Use Linear Regression
y_train_pred, y_test_pred, _, importances, _ = train_linear_regression(
    X_train, X_test, y_train, y_test
)

# === Export to Excel (with index & column names) ===
output_file = f"metrics_{target}_LR.xlsx"

calc_metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    importances.to_excel(writer, sheet_name="Importances")      # preserves index & cols
    calc_metrics.to_excel(writer, sheet_name="Calc Metrics")  # already a DataFrame

# === Plot ===
plot_predictions_vs_actual(
    y_train, y_train_pred,
    y_test, y_test_pred
)