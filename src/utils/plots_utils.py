import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# Function to create scatter plot with regression
def create_scatter_regression(x, y, ax, title, xlabel, ylabel, binned=False, num_bins=20, split_regression_at=None, subgroups=None):
    """
    Creates a scatter plot with a regression line, with an option for binned scattering
    and split regression lines.
    
    Parameters:
    -----------
    x : pd.Series
        X-axis data
    y : pd.Series
        Y-axis data
    ax : matplotlib.axes.Axes
        The axes to plot on
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    binned : bool, optional
        Whether to use binned scatter plot
    num_bins : int, optional
        Number of bins for binned scatter plot
    split_regression_at : float, optional
        Value to split regression line at
    subgroups : dict, optional
        Dictionary containing subgroup masks and labels for multiple regression lines
        Format: {'label': (mask, color)}
    """
    # Drop NA values together
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
        if title:  # Only set title if it's not empty
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    # Handle multiple subgroups if provided
    if subgroups is not None:
        # Plot scatter points/bins for each subgroup
        for label, (subgroup_mask, color) in subgroups.items():
            if subgroup_mask.sum() > 0:  # Only plot if there are points
                x_subgroup = x_clean[subgroup_mask]
                y_subgroup = y_clean[subgroup_mask]
                
                if binned and x_subgroup.nunique() > 1:
                    df_temp = pd.DataFrame({'x': x_subgroup, 'y': y_subgroup})
                    df_temp['bin'] = pd.cut(df_temp['x'], bins=num_bins, include_lowest=True, duplicates='drop')
                    binned_data = df_temp.groupby('bin', observed=True).agg(
                        x_mean=('x', 'mean'),
                        y_mean=('y', 'mean')
                    ).dropna()
                    ax.scatter(binned_data['x_mean'], binned_data['y_mean'], 
                             color=color, alpha=0.7, label=f'{label} (Binned)')
                else:
                    ax.scatter(x_subgroup, y_subgroup, color=color, alpha=0.3, label=label)

                # Fit & plot regression line for this subgroup
                if subgroup_mask.sum() > 1:  # Only fit regression if enough points
                    m, b = np.polyfit(x_subgroup, y_subgroup, 1)
                    xs = np.linspace(x_subgroup.min(), x_subgroup.max(), 100)
                    ax.plot(xs, m * xs + b, color=color, linestyle='--', 
                           label=f'{label} (Slope: {m:.3f})')

    # Handle split regression if provided
    elif split_regression_at is not None:
        # Split data
        mask_neg = x_clean < split_regression_at
        mask_pos = x_clean >= split_regression_at

        # Plot scatter points/bins
        if binned and x_clean.nunique() > 1:
            df_temp = pd.DataFrame({'x': x_clean, 'y': y_clean})
            df_temp['bin'] = pd.cut(df_temp['x'], bins=num_bins, include_lowest=True, duplicates='drop')
            binned_data = df_temp.groupby('bin', observed=True).agg(
                x_mean=('x', 'mean'),
                y_mean=('y', 'mean')
            ).dropna()
            ax.scatter(binned_data['x_mean'], binned_data['y_mean'], alpha=0.7, label='Binned Average')
        else:
            ax.scatter(x_clean, y_clean, alpha=0.3, label='Data points')

        # Fit & plot regression for x < split_regression_at
        if mask_neg.sum() > 1:
            m_neg, b_neg = np.polyfit(x_clean[mask_neg], y_clean[mask_neg], 1)
            xs_neg = np.linspace(x_clean[mask_neg].min(), x_clean[mask_neg].max(), 100)
            ax.plot(xs_neg, m_neg * xs_neg + b_neg, color='red', linestyle='--', 
                   label=f'Slope (< {split_regression_at}): {m_neg:.3f}')

        # Fit & plot regression for x >= split_regression_at
        if mask_pos.sum() > 1:
            m_pos, b_pos = np.polyfit(x_clean[mask_pos], y_clean[mask_pos], 1)
            xs_pos = np.linspace(x_clean[mask_pos].min(), x_clean[mask_pos].max(), 100)
            ax.plot(xs_pos, m_pos * xs_pos + b_pos, color='green', linestyle='--', 
                   label=f'Slope (>= {split_regression_at}): {m_pos:.3f}')
        
        ax.axvline(x=split_regression_at, color='gray', linestyle=':')

    else:
        # Single regression line calculated on original data
        if binned and x_clean.nunique() > 1:
            df_temp = pd.DataFrame({'x': x_clean, 'y': y_clean})
            df_temp['bin'] = pd.cut(df_temp['x'], bins=num_bins, include_lowest=True, duplicates='drop')
            binned_data = df_temp.groupby('bin', observed=True).agg(
                x_mean=('x', 'mean'),
                y_mean=('y', 'mean')
            ).dropna()
            ax.scatter(binned_data['x_mean'], binned_data['y_mean'], alpha=0.7, label='Binned Average')
        else:
            ax.scatter(x_clean, y_clean, alpha=0.3, label='Data points')

        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        line_x = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(line_x, p(line_x), "r--", alpha=0.8, label=f'Slope: {z[0]:.3f}')

    if title:  # Only set title if it's not empty
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

def get_project_root():
    """Get the project root directory."""
    return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def save_figure(fig, filename, file_path):
    """Save a matplotlib figure to the figures directory."""
    figures_dir = os.path.join(get_project_root(), file_path)
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, filename))
    plt.close(fig)

def create_forward_fx_bins(forward_fx_values, n_bins=10):
    """Create equal-width bins for forward FX values"""
    p1 = np.percentile(forward_fx_values, 1)
    p99 = np.percentile(forward_fx_values, 99)
    min_forward_fx = p1
    max_forward_fx = p99
    bin_width = (max_forward_fx - min_forward_fx) / n_bins
    
    bin_edges = [min_forward_fx + i * bin_width for i in range(n_bins + 1)]
    bin_labels = []
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == n_bins - 1:  # Last bin - include upper bound
            bin_labels.append(f"[{lower:.3f}, {upper:.3f}]")
        else:
            bin_labels.append(f"[{lower:.3f}, {upper:.3f})")
    
    return bin_edges, bin_labels

def assign_forward_fx_bin(forward_fx, bin_edges, bin_labels):
    """Assign forward FX value to appropriate bin"""
    if pd.isna(forward_fx):
        return None
    
    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:  # Last bin - include upper bound
            if bin_edges[i] <= forward_fx <= bin_edges[i + 1]:
                return bin_labels[i]
        else:
            if bin_edges[i] <= forward_fx < bin_edges[i + 1]:
                return bin_labels[i]
    
    return None