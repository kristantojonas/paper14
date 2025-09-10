import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# === Load Excel ===
df = pd.read_excel("IMPORTANCES.xlsx", sheet_name="Sheet2")

# Convert "#N/A" to NaN and drop them
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

# === Count frequency of each parameter ===
param_counts = Counter(df["Parameters"])
most_common_params = [p for p, _ in param_counts.most_common(6)]  # top 6 frequent

# === Assign contrasting colors to most common ===
highlight_colors = [
    "#e41a1c",  # bright red
    "#377eb8",  # strong blue
    "#4daf4a",  # vivid green
    "#ff7f00",  # bright orange
    "#984ea3",  # purple
    "#00ced1"   # cyan
]

muted_colors = [
    "#7f7f7f", "#a6cee3", "#b2df8a",
    "#fdbf6f", "#cab2d6", "#bc80bd", "#8c6d31"
]

property_colors = {}
for i, p in enumerate(most_common_params):
    property_colors[p] = highlight_colors[i % len(highlight_colors)]

# Assign muted colors for the rest
for p in df["Parameters"].unique():
    if p not in property_colors:
        property_colors[p] = muted_colors[len(property_colors) % len(muted_colors)]

# === Setup subplots ===
fig, axes = plt.subplots(2, 3, figsize=(11, 6),dpi=150)

models = ["RF", "GB"]
products = ["Solid", "Liquid", "Gas"]

# Prepare legend handles once
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=property_colors[p],
               markersize=8, label=p)
    for p in property_colors.keys()
]

for i, model in enumerate(models):
    for j, product in enumerate(products):
        ax = axes[i, j]

        # Filter data
        subset = df[(df["Model"] == model) & (df["Product"] == product)]
        values = subset["Value"].values
        labels = subset["Parameters"].values
        colors = [property_colors[p] for p in labels]

        # Pie chart
        ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct="%.1f%%",
            textprops={"color": "white", "fontsize": 8}
        )

        # Add product label in center
        ax.text(0, 0, product,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

# Global legend on the right
fig.legend(handles=legend_handles,
           loc="right",
           bbox_to_anchor=(1.0, 0.5),
           title="Parameters")

# Adjust spacing
fig.subplots_adjust(left=0.0, right=0.8, top=1, bottom=0.0,
                    wspace=0.0, hspace=0.0)

plt.show()
