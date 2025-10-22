import pandas as pd
import numpy as np
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

# Prepare subplot grid for PDP
fig_pdp, axes_pdp = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True, dpi=75)
model_rows = ['rf', 'gbt']
targets = ['Solid', 'Liquid', 'Gas']

# =====================================================
# Store importances and max predicted yield
# =====================================================
importances_summary = { (model_type, target): [] for model_type in model_rows for target in targets }
max_yield = 0
predictions_store = {}

# =====================================================
# Train models under 50 random states
# =====================================================
for row_idx, model_type in enumerate(model_rows):
    for col_idx, target in enumerate(targets):
        NE = CONFIG[target][model_type]["NE"]
        MD = CONFIG[target][model_type]["MD"]

        all_importances = []

        for rs in range(50):
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

        # Save importances to Excel
        imp_df = pd.concat(all_importances, axis=1)
        imp_df.columns = [f"RS_{i}" for i in range(imp_df.shape[1])]
        imp_df.to_excel(f"FeatureImportance_{target}_{model_type}.xlsx")

        # Compute mean importances to select top 2 features
        mean_importances = imp_df.mean(axis=1)
        top_features = mean_importances.sort_values(ascending=False).index[:2]
        f1, f2 = top_features
        importances_summary[(model_type, target)] = mean_importances

        # Use last random state to generate PDP
        n_points = 101
        f1_range = np.linspace(X_train[f1].min(), X_train[f1].max(), n_points)
        f2_range = np.linspace(X_train[f2].min(), X_train[f2].max(), n_points)
        xx, yy = np.meshgrid(f1_range, f2_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Preserve original X_train
        X_train_orig = X_train.copy()
        X_ref = X_train.mean().to_dict()
        X_pdp = pd.DataFrame([{**X_ref, f1: x1, f2: x2} for x1, x2 in grid_points])
        X_pdp = X_pdp[X_train.columns]

        zz = model.predict(X_pdp).reshape(xx.shape)
        max_yield = max(max_yield, zz.max())

        # Restore X_train
        X_train = X_train_orig.copy()
        predictions_store[(row_idx, col_idx)] = (zz, f1, f2, X_train)

# =====================================================
# Plot PDPs with unified color scale
# =====================================================
for (row_idx, col_idx), (zz, f1, f2, X_train) in predictions_store.items():
    ax = axes_pdp[row_idx, col_idx]

    # Axis ranges match original feature values
    f1_min, f1_max = df_raw[f1].min(), df_raw[f1].max()
    f2_min, f2_max = df_raw[f2].min(), df_raw[f2].max()
    
    im = ax.imshow(
        zz, origin='lower', aspect='auto',
        extent=[f1_min, f1_max, f2_min, f2_max],  # now uses original feature range
        cmap='jet', vmin=0, vmax=max_yield
    )
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)

# Single colorbar for all PDPs
fig_pdp.colorbar(im, ax=axes_pdp, orientation='vertical', fraction=0.02, pad=0.04, label="Yield prediction")

axes_pdp[0, 0].text(-0.15, 1.05, "(a)", transform=axes_pdp[0, 0].transAxes,
                    fontsize=13, fontweight='bold', va='top', ha='right')
axes_pdp[1, 0].text(-0.15, 1.05, "(b)", transform=axes_pdp[1, 0].transAxes,
                    fontsize=13, fontweight='bold', va='top', ha='right')

plt.savefig("PDP_Grid_RF_GBT_OneCB_RS0_50.png", dpi=400)
plt.close()
print("🎯 PDP grid saved.")

# =====================================================
# High-contrast significance pie charts with "Others" (importance >5%)
# =====================================================
rows, cols = [], []
values, model_labels, product_labels = [], [], []

for (model_type, target), mean_imp in importances_summary.items():
    for param, val in mean_imp.items():
        rows.append(param)
        cols.append(f"{target}_{model_type}")
        values.append(val)
        model_labels.append("RF" if model_type=="rf" else "GB")
        product_labels.append(target)

df_sig = pd.DataFrame({
    "Parameters": rows,
    "Model": model_labels,
    "Product": product_labels,
    "Value": values
})

# Top 10 color palette
highlight_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
                    "#984ea3", "#00ced1", "#f781bf", "#a65628",
                    "#999999", "#66c2a5"]
others_color = "#000000"

all_params = df_sig.groupby("Parameters")["Value"].mean()
top_params = all_params.sort_values(ascending=False).index[:10]

property_colors = {p: highlight_colors[i] for i, p in enumerate(top_params)}
property_colors["Others"] = others_color

fig_pie, axes_pie = plt.subplots(2, 3, figsize=(11,6), dpi=150)
models = ["RF","GB"]
products = ["Solid","Liquid","Gas"]

for i, model in enumerate(models):
    for j, product in enumerate(products):
        ax = axes_pie[i,j]
        subset = df_sig[(df_sig["Model"]==model) & (df_sig["Product"]==product)]
        subset = subset.groupby("Parameters")["Value"].mean()
        subset = subset / subset.sum() * 100  # convert to %

        significant = subset[subset > 5]
        others = subset[subset <= 5].sum()
        pie_values = significant.tolist() + [others]
        pie_labels = significant.index.tolist() + ["Others"]

        colors = [property_colors.get(p, others_color) for p in pie_labels]

        ax.pie(pie_values, labels=None, colors=colors,
               autopct="%.1f%%", textprops={"color":"white","fontsize":8})
        ax.text(0,0,product, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

legend_handles = [plt.Line2D([0],[0], marker="o", color="w",
                             markerfacecolor=property_colors[p],
                             markersize=8, label=p)
                  for p in list(top_params) + ["Others"]]

fig_pie.legend(handles=legend_handles, loc="right", bbox_to_anchor=(1.0,0.5), title="Parameters")
fig_pie.subplots_adjust(left=0.0, right=0.8, top=1, bottom=0.0, wspace=0.0, hspace=0.0)
fig_pie.savefig("FeatureSignificance_PieCharts_HighContrast_10Colors.png", dpi=400, bbox_inches='tight')
plt.close(fig_pie)
print("🎯 High-contrast pie charts saved with top parameters >5%, others grouped (10 colors + black).")
