import pandas as pd
from SUB_ML import train_random_forest
from SUB_STAT_METRICS import calculate_metrics
from SUB_PREPROCESSING import prepare_data_wgf
from SUB_PLOTTING import plot_predictions_vs_actual
from SUB_PREPREPROCESSING import PREPREPROCESSING

# === Load data ===
y_cols = ['Gas', 'Liquid', 'Solid']
X_ops=['Particle Size (mm)','Temperature (C)','Residence Time (h)',
       'Carrier Gas (mL/min)','Heating Rate (C/h)']
df_raw = pd.read_excel('RAW ML.xlsx', skiprows=1)
df = df_raw[df_raw.columns.difference(['No.', 'First Author'])]#,'Family','Genus'])]
df = PREPREPROCESSING(df)

# === Redevelop model using best config ===
target = 'Solid'
best_rs = 14
best_MD  = 17
best_NE  = 21
# target = 'Liquid'
# best_rs = 29
# best_MD  = 19
# best_NE  = 23
# target = 'Gas'
# best_rs = 44
# best_MD  = 20
# best_NE  = 20

X_train, X_test, y_train, y_test = prepare_data_wgf(
    df, target, y_cols, best_rs,
    exclude_hier_cols=X_ops, test_size=0.25
)

y_train_pred, y_test_pred, _, metrics_df, importances = train_random_forest(
    X_train, X_test, y_train, y_test, best_NE, best_MD
)

# === Export to Excel (with index & column names) ===
output_file = f"wgf_metrics_{target}_RF.xlsx"

calc_metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="Metrics DF")      # preserves index & cols
    calc_metrics.to_excel(writer, sheet_name="Calc Metrics")  # already a DataFrame

# === Plot ===
plot_predictions_vs_actual(
    y_train, y_train_pred,
    y_test, y_test_pred
)
