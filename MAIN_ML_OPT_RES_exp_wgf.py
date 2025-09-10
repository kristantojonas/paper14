import pandas as pd
import numpy as np
from SUB_STAT_METRICS import calculate_metrics
from SUB_PREPROCESSING import prepare_data_wgf
from SUB_ML import train_random_forest, train_gbt, train_linear_regression
from SUB_PREPREPROCESSING import PREPREPROCESSING
import matplotlib
matplotlib.use('Agg')  # ensures plots are not displayed

def model_full_search_to_excel(df, target, y_cols, exclude_hier_cols,
                               MD, NE, rand_states,
                               test_size=0.25,
                               model_type="rf",
                               filename="results_all.xlsx"):
    """
    Run search for specific target with specific random states and hyperparams.
    """
    results = []
    best_config = None

    for rs in rand_states:
        X_train, X_test, y_train, y_test = prepare_data_wgf(
            df, target, y_cols, rs,
            exclude_hier_cols=exclude_hier_cols, test_size=test_size
        )

        # ============ LINEAR REGRESSION ============
        if model_type == "lr":
            y_train_pred, y_test_pred, _, importances, _ = train_linear_regression(
                X_train, X_test, y_train, y_test
            )
            metrics_df = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)

            row = {
                "Target": target,
                "rand_state": rs,
                "MD": None,
                "NE": None,
                "Train MSE": metrics_df.loc["train", "MSE"],
                "Train R2":  metrics_df.loc["train", "R2"],
                "Test MSE":  metrics_df.loc["test", "MSE"],
                "Test R2":   metrics_df.loc["test", "R2"]
            }
            results.append(row)
            save_realtime(filename, model_type, results)
            continue

        # ============ RF / GBT ============
        for md in MD:
            for ne in NE:
                if model_type == "rf":
                    y_train_pred, y_test_pred, _, importances, _ = train_random_forest(
                        X_train, X_test, y_train, y_test, n_est=ne, max_de=md
                    )
                elif model_type == "gbt":
                    y_train_pred, y_test_pred, _, importances, _ = train_gbt(
                        X_train, X_test, y_train, y_test, n_est=ne, max_de=md
                    )
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                metrics_df = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)

                row = {
                    "Target": target,
                    "rand_state": rs,
                    "MD": md,
                    "NE": ne,
                    "Train MSE": metrics_df.loc["train", "MSE"],
                    "Train R2":  metrics_df.loc["train", "R2"],
                    "Test MSE":  metrics_df.loc["test", "MSE"],
                    "Test R2":   metrics_df.loc["test", "R2"]
                }
                results.append(row)
                save_realtime(filename, model_type, results)

    return pd.DataFrame(results), best_config


def save_realtime(filename, sheet_name, results):
    """Helper to write results incrementally into Excel (per model sheet)."""
    results_df = pd.DataFrame(results)
    try:
        with pd.ExcelWriter(filename, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(filename, mode="w", engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)


# =====================
# MAIN SCRIPT
# =====================
y_cols = ['Liquid','Gas','Solid']
X_ops = ['Particle Size (mm)', 'Temperature (C)', 'Residence Time (h)',
         'Carrier Gas (mL/min)', 'Heating Rate (C/h)']

# Load data
df_raw = pd.read_excel('RAW ML.xlsx', skiprows=1)
df = df_raw[df_raw.columns.difference(['No.', 'First Author','Family','Genus'])]
df = PREPREPROCESSING(df)

# Custom configs per target
configs = {
    "Gas": {
        "rand_states": [33],#[44],
        "MD": np.geomspace(1, 50, 10, dtype=int),
        "NE": np.geomspace(1, 100, 10, dtype=int)
    },
    "Liquid": {
        "rand_states": [33],#[17],
        "MD": np.geomspace(1, 50, 10, dtype=int),
        "NE": np.geomspace(1, 100, 10, dtype=int)
    },
    "Solid": {
        "rand_states": [33],#[14],
        "MD": np.geomspace(1, 50, 10, dtype=int),
        "NE": np.geomspace(1, 100, 10, dtype=int)
    }
}

# Run automatically for each target and model
for target, cfg in configs.items():
    for model in ["rf", "gbt"]:
        filename = f"wgf_exp_{model}_results_{target}.xlsx"
        results, best = model_full_search_to_excel(
            df, target=target, y_cols=y_cols, exclude_hier_cols=X_ops,
            MD=cfg["MD"], NE=cfg["NE"],
            rand_states=cfg["rand_states"],
            model_type=model,
            filename=filename
        )
