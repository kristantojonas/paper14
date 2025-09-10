import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions_vs_actual(y_train, y_train_pred, y_test, y_test_pred, graph_title=None):
    n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
    y_train_df = y_train if isinstance(y_train, pd.DataFrame) else pd.DataFrame(y_train)
    y_train_pred_df = y_train_pred if isinstance(y_train_pred, pd.DataFrame) else pd.DataFrame(y_train_pred)
    y_test_df = y_test if isinstance(y_test, pd.DataFrame) else pd.DataFrame(y_test)
    y_test_pred_df = y_test_pred if isinstance(y_test_pred, pd.DataFrame) else pd.DataFrame(y_test_pred)

    if n_outputs == 1:
        y_train_df.columns = ['Output']
        y_train_pred_df.columns = ['Output']
        y_test_df.columns = ['Output']
        y_test_pred_df.columns = ['Output']
    
    n_cols = y_train_df.shape[1]
    fig, axes = plt.subplots(1, n_cols, figsize=(3.25 * n_cols, 3), squeeze=False, dpi=300)

    for i, col in enumerate(y_train_df.columns):
        ax = axes[0, i]
        min_val = 0
        max_val = 100
        margin = 0.1 * (max_val - min_val)
        lower = min_val - margin
        upper = max_val + margin

        ax.scatter(y_train_df[col], y_train_pred_df[col], color='b', alpha=0.5, label='Train', s=7.5)
        ax.scatter(y_test_df[col], y_test_pred_df[col], color='r', alpha=0.5, label='Test', s=7.5)

        ax.plot([lower, upper], [lower, upper], 'k', label='Diagonal', linewidth=.75)
        ax.plot([lower, upper], [lower * 0.9, upper * 0.9], 'g--', label='±10%', linewidth=.75)
        ax.plot([lower, upper], [lower * 1.1, upper * 1.1], 'g--', linewidth=.75)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig
