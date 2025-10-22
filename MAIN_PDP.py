import pandas as pd
import numpy as np
from SUB_STAT_METRICS import calculate_metrics
from SUB_PREPROCESSING import prepare_data
from SUB_ML import train_random_forest, train_gbt
from SUB_PREPREPROCESSING import PREPREPROCESSING
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================
# PDP-compatible model training
# =====================================================
def train_rf_for_pdp(X_train, y_train, n_est=26, max_de=121):
    model = RandomForestRegressor(n_estimators=n_est, max_depth=max_de)
    model.fit(X_train, y_train)
    return model

def train_gbt_for_pdp(X_train, y_train, n_est=26, max_de=121):
    model = GradientBoostingRegressor(n_estimators=n_est, max_depth=max_de)
    model.fit(X_train, y_train)
    return model

# =====================================================
# CONFIGURATION
# =====================================================
CONFIG = {
    "Solid": {"rf": {"NE": 17, "MD": 21}, "gbt": {"NE": 16, "MD": 38}},
    "Liquid": {"rf": {"NE": 19, "MD": 23}, "gbt": {"NE": 15, "MD": 32}},
    "Gas": {"rf": {"NE": 20, "MD": 20}, "gbt": {"NE": 17, "MD": 34}},
}

y_cols = ['Liquid', 'Gas', 'Solid']
X_ops = ['Particle Size (mm)', 'Temperature (°C)', 'Residence Time (h)',
         'Carrier Gas (mL/min)', 'Heating Rate (C/h)']

# =====================================================
# MAIN SCRIPT
# =====================================================
print("\n🚀 Starting PDP model training and generation...\n")

# Load and preprocess data
df_raw = pd.read_excel('RAW ML.xlsx', skiprows=1)
df = df_raw[df_raw.columns.difference(['No.', 'First Author'])]
df = PREPREPROCESSING(df)

# Prepare subplot grid
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True, dpi=75)
model_rows = ['rf']#, 'gbt']
targets = ['Solid']#, 'Liquid', 'Gas']

# =====================================================
# Store importances and max predicted yield
# =====================================================
importances_summary = { (model_type, target): [] for model_type in model_rows for target in targets }
max_yield = 0
predictions_store = {}

# =====================================================
# Train models under random states 0-50
# =====================================================
for row_idx, model_type in enumerate(model_rows):
    for col_idx, target in enumerate(targets):
        NE = CONFIG[target][model_type]["NE"]
        MD = CONFIG[target][model_type]["MD"]
        
        all_importances = []

        for rs in range(50):  # Random states 0 to 50
            X_train, X_test, y_train, y_test = prepare_data(
                df, target, y_cols, rs,
                exclude_hier_cols=X_ops, test_size=0.25
            )

            # Train model and get feature importances
            if model_type == "rf":
                _, _, _, importances, _ = train_random_forest(
                    X_train, X_test, y_train, y_test, n_est=NE, max_de=MD
                )
                model = train_rf_for_pdp(X_train, y_train[target], n_est=NE, max_de=MD)
            else:
                _, _, _, importances, _ = train_gbt(
                    X_train, X_test, y_train, y_test, n_est=NE, max_de=MD
                )
                model = train_gbt_for_pdp(X_train, y_train[target], n_est=NE, max_de=MD)

            # Store feature importances
            if isinstance(importances, pd.DataFrame):
                imp_values = importances.values.flatten()
                imp_index = importances.index
            else:
                imp_values = np.array(importances)
                imp_index = X_train.columns

            all_importances.append(pd.Series(imp_values, index=imp_index))

        # Save all importances to Excel
        imp_df = pd.concat(all_importances, axis=1)
        imp_df.columns = [f"RS_{i}" for i in range(imp_df.shape[1])]
        imp_df.to_excel(f"FeatureImportance_{target}_{model_type}.xlsx")

        # Compute mean importances to select top 2 features
        mean_importances = imp_df.mean(axis=1)
        top_features = mean_importances.sort_values(ascending=False).index[:2]
        f1, f2 = top_features

        # Create PDP grid based on top 2 features (use last random state)
        n_points = 101
        f1_range = np.linspace(X_train[f1].min(), X_train[f1].max(), n_points)
        f2_range = np.linspace(X_train[f2].min(), X_train[f2].max(), n_points)
        xx, yy = np.meshgrid(f1_range, f2_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        X_ref = X_train.mean().to_dict()
        X_pdp = pd.DataFrame([{**X_ref, f1: x1, f2: x2} for x1, x2 in grid_points])
        X_pdp = X_pdp[X_train.columns]

        zz = model.predict(X_pdp).reshape(xx.shape)
        max_yield = max(max_yield, zz.max())

        predictions_store[(row_idx, col_idx)] = (zz, f1, f2, X_train)

# =====================================================
# Plot PDPs with unified color scale
# =====================================================
for (row_idx, col_idx), (zz, f1, f2, X_train) in predictions_store.items():
    ax = axes[row_idx, col_idx]

    f1_min, f1_max = X_train[f1].min(), X_train[f1].max()
    f2_min, f2_max = X_train[f2].min(), X_train[f2].max()

    im = ax.imshow(
        zz, origin='lower', aspect='auto',
        extent=[f1_min, f1_max, f2_min, f2_max],
        cmap='cividis', vmin=0, vmax=max_yield
    )
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)

# Single colorbar on right
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label="Yield prediction")

# Row labels
axes[0, 0].text(-0.15, 1.05, "(a)", transform=axes[0, 0].transAxes,
                fontsize=13, fontweight='bold', va='top', ha='right')
axes[1, 0].text(-0.15, 1.05, "(b)", transform=axes[1, 0].transAxes,
                fontsize=13, fontweight='bold', va='top', ha='right')

# Save figure
plt.savefig("PDP_Grid_RF_GBT_OneCB_RS0_50.png", dpi=400)
plt.close()
print("\n🎯 PDP grid saved. Feature importances saved to Excel for all random states.")
