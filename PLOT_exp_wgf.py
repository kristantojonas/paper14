import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})

def get_maxima(data, x):
    """Return maxima dataframe per x and Type."""
    maxima = []
    order = sorted(data[x].unique())
    for val in order:
        for t in ["Train R2", "Test R2"]:
            subset = data[(data[x] == val) & (data["Type"] == t)]["R2"]
            if len(subset) > 0:
                maxima.append((val, t, subset.max()))
    return pd.DataFrame(maxima, columns=[x, "Type", "R2"]), order


def plot_trend(ax, data, x):
    """Plot max trend lines on given ax."""
    maxima_df, order = get_maxima(data, x)
    
    # Plot lines
    for t, color, simple_label in zip(
        ["Train R2", "Test R2"],
        ["blue", "red"],
        ["Train$_{max}$", "Test$_{max}$"]
    ):
        subset = maxima_df[maxima_df["Type"] == t]
        ax.plot(subset[x], subset["R2"], marker="o", color=color, label=simple_label)
    
    # Styling
    ax.set_ylim(-.05, 1.05)
    ax.set_xticks(order)
    ax.set_xticklabels([val if i % 2 == 0 else "" for i, val in enumerate(order)])
    # no legend here


def make_model_plot(model):
    """Create one 3x2 figure for a given model."""
    targets = ["Solid", "Liquid", "Gas"]
    letters = ["a", "b", "c"]

    fig, axes = plt.subplots(3, 2, figsize=(7.1, 10), dpi=75, sharey=True)

    for i, target in enumerate(targets):
        file = f"wgf_exp_{model}_results_{target}.xlsx"
        
        if not os.path.exists(file):
            print(f"⚠️ Skipped: {file} not found")
            continue
        
        # Read excel
        df = pd.read_excel(file)
        
        # Reshape into long format
        df_long = df.melt(
            id_vars=["MD", "NE"], 
            value_vars=["Train R2", "Test R2"],
            var_name="Type", 
            value_name="R2"
        )
        
        # Left column = NE, Right column = MD
        ax_ne = axes[i, 0]
        ax_md = axes[i, 1]
        
        plot_trend(ax_ne, df_long, "NE")
        plot_trend(ax_md, df_long, "MD")
        
        # Add row label (a, b, c) at top left
        ax_ne.text(0.075, .975, letters[i], transform=ax_ne.transAxes,
                   fontsize=12, fontweight="bold", va="top", ha="right")

    # Shared labels
    fig.supylabel("R$^2$", fontsize=12,fontweight="bold")
    axes[-1, 0].set_xlabel("NE",fontweight="bold")
    axes[-1, 1].set_xlabel("MD",fontweight="bold")
    
    # Put legend inside bottom-right subplot (Gas–MD)
    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[-1, 1].legend(handles, labels, loc="lower right", fontsize=12)

    plt.subplots_adjust(
        hspace=0, wspace=0,
        left=0.1, right=0.975, top=.975, bottom=0.05
    )
    
    plt.savefig(f"wgf_exp_{model}_all_targets_trends.png", dpi=150, bbox_inches="tight")
    plt.show()


# === Generate plots for both models ===
make_model_plot("rf")
make_model_plot("gbt")
