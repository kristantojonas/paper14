import pandas as pd
import numpy as np
from SUB_STAT_METRICS import calculate_metrics
from SUB_PREPROCESSING import prepare_data
from SUB_ML import train_random_forest, train_gbt, train_linear_regression
from SUB_PLOTTING import plot_predictions_vs_actual
from SUB_PREPREPROCESSING import PREPREPROCESSING
import matplotlib
matplotlib.use('Agg')  # ensures plots are not displayed

def model_full_search_to_excel(df, target, y_cols, exclude_hier_cols,
                               MD=[5, 10, 20, None], NE=[50, 100, 200],
                               n_states=10, test_size=0.25,
                               model_type="rf",
                               filename="results_all.xlsx"):
    """
    Run full grid search with multiple random states for RF, GBT, or Linear Regression,
    export results to Excel in real-time, and overwrite the best config’s
    plot + importances + metrics Excel if a better Test R2 is found
    AND Train R2 > Test R2.
    """
    results = []
    best_test_r2 = -np.inf
    best_config = None

    for rs in range(n_states):
        X_train, X_test, y_train, y_test = prepare_data(
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

            # ✅ Only save best if Test R2 improves AND Train R2 > Test R2
            if (row["Test R2"] > best_test_r2) and (row["Train R2"] > row["Test R2"]):
                best_test_r2 = row["Test R2"]
                best_config = {"rand_state": rs, "MD": None, "NE": None}
                export_best(target, model_type, X_train, X_test, y_train, y_train_pred,
                            y_test, y_test_pred, importances, metrics_df, best_config)

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

                # ✅ Only save best if Test R2 improves AND Train R2 > Test R2
                if (row["Test R2"] > best_test_r2) and (row["Train R2"] > row["Test R2"]):
                    best_test_r2 = row["Test R2"]
                    best_config = {"rand_state": rs, "MD": md, "NE": ne}
                    export_best(target, model_type, X_train, X_test, y_train, y_train_pred,
                                y_test, y_test_pred, importances, metrics_df, best_config)

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


def export_best(target, model_type, X_train, X_test, y_train, y_train_pred,
                y_test, y_test_pred, importances, metrics_df, best_config):
    """Export best plot + Excel (metrics, importances, best_config)."""
    output_file = f"best_{model_type}_{target}.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Metrics
        metrics_df.to_excel(writer, sheet_name="Metrics")
        # Importances
        if importances is not None:
            importances.to_excel(writer, sheet_name="Importances")
        # Best config
        pd.DataFrame([best_config]).to_excel(writer, sheet_name="BestConfig", index=False)

    # Save plot (do not show)
    fig = plot_predictions_vs_actual(y_train, y_train_pred, y_test, y_test_pred,
                                     graph_title=f"{model_type.upper()} - {target}")
    plot_filename = f"plot_best_{model_type}_{target}.png"
    fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    fig.clf()  # clear figure to prevent memory issues
    # print(f"🏆 Updated best export: {output_file}, {plot_filename}")


# =====================
# MAIN SCRIPT
# =====================
y_cols = ['Liquid','Gas','Solid']
y_col_now = ['Liquid','Gas','Solid']
X_ops = ['Particle Size (mm)', 'Temperature (C)', 'Residence Time (h)',
         'Carrier Gas (mL/min)', 'Heating Rate (C/h)']

# Load data
df_raw = pd.read_excel('RAW ML.xlsx', skiprows=1)
df = df_raw[df_raw.columns.difference(['No.', 'First Author'])]
df = PREPREPROCESSING(df)

# Run automatically for each target and model
for target in y_col_now:
    for model in ["lr","rf","gbt"]:  # can extend to ["rf", "gbt", "lr"]
        filename = f"{model}_results_{target}.xlsx"
        # print(f"\n🚀 Running {model.upper()} search for target = {target} ...")
        results, best = model_full_search_to_excel(
            df, target=target, y_cols=y_cols, exclude_hier_cols=X_ops,
            MD=np.arange(15,25),
            NE=np.arange(20,40),
            n_states=45, model_type=model, filename=filename
        )
        # print(f"Best config for {target}: {best}")