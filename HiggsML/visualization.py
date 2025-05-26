import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn for nice plot quicker
import os
import logging

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

def correlation_plots(dfall ,target,columns=None):
    """
    Plots correlation matrices of the dataset features.

    Args:
    * columns (list): The list of column names to consider (default: None, which includes all columns).

    .. Image:: images/correlation_plots.png
    """
    
    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")
    
    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")

    caption = ["Signal feature", "Background feature"]

    for i in range(2):

        dfplot = pd.DataFrame(dfall, columns=columns)

        print(caption[i], " correlation matrix")
        corrMatrix = dfplot[target == i].corr()
        sns.heatmap(corrMatrix, annot=True)
        plt.title("Correlation matrix of features")
        plt.show()

def pair_plots(dfall ,target,sample_size=10, columns=None):
    """
    Plots pair plots of the dataset features.

    Args:
        * sample_size (int): The number of samples to consider (default: 10).
        * columns (list): The list of column names to consider (default: None, which includes all columns).

    .. Image:: images/pair_plot.png
    """
    if columns is None:
        columns = []
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")
    
    df_sample = dfall[columns].copy()
    df_sample["Label"] = target

    df_sample_S = df_sample[target == 1].sample(n=sample_size)
    df_sample_B = df_sample[target == 0].sample(n=sample_size)
    frames = [df_sample_S, df_sample_B]
    del df_sample
    df_sample = pd.concat(frames)

    sns.set_theme(rc={"figure.figsize": (16, 14)}, style="whitegrid")

    ax = sns.PairGrid(df_sample, hue="Label")
    ax.map_upper(sns.scatterplot, alpha=0.5, size=0.3)
    ax.map_lower(
        sns.kdeplot, fill=True, levels=5, alpha=0.5
    )  # Change alpha value here
    ax.map_diag(
        sns.histplot,
        alpha=0.5,
        bins=20,
    )  # Change alpha value here
    ax.add_legend(title="Legend", labels=["Signal", "Background"], fontsize=12)

    legend = ax.legend
    for line in legend.get_lines():  # For lines
        line.set_alpha(0.5)
        line.set_linewidth(1.5)

    plt.rcParams["figure.facecolor"] = "w"  # Set the figure facecolor to white
    ax.figure.suptitle("Pair plots of features")
    plt.show()
    plt.close()

def stacked_histogram(dfall, target, weights, detailed_label, field_name, mu_hat=1.0, nbins=30,y_scale='linear'):
    """
    Plots a stacked histogram of a specific field in the dataset.

    Args:
        * field_name (str): The name of the field to plot.
        * mu_hat (float): The value of mu (default: 1.0).
        * bins (int): The number of bins for the histogram (default: 30).

    .. Image:: images/stacked_histogram.png
    """    
    field = dfall[field_name]

    weight_keys = {}
    keys = np.unique(detailed_label)

    for key in keys:
        weight_keys[key] = weights[detailed_label == key]    

    print("keys" , keys)
    print("keys 2", weight_keys.keys())

    sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")

    lower_percentile = 0
    upper_percentile = 97.5

    lower_bound = np.percentile(field, lower_percentile)
    upper_bound = np.percentile(field, upper_percentile)

    field_clipped = field[(field >= lower_bound) & (field <= upper_bound)]
    weights_clipped = weights[
        (field >= lower_bound) & (field <= upper_bound)
    ]
    target_clipped = target[
        (field >= lower_bound) & (field <= upper_bound)
    ]
    detailed_labels_clipped = detailed_label[
        (field >= lower_bound) & (field <= upper_bound)
    ]

    min_value = field_clipped.min()
    max_value = field_clipped.max()

    # Define the bin edges
    bins = np.linspace(min_value, max_value, nbins + 1)

    hist_s, bins = np.histogram(
        field_clipped[target_clipped == 1],
        bins=bins,
        weights=weights_clipped[target_clipped == 1],
    )

    hist_b, bins = np.histogram(
        field_clipped[target_clipped == 0],
        bins=bins,
        weights=weights_clipped[target_clipped == 0],
    )

    hist_bkg = hist_b.copy()

    higgs = "htautau"

    for key in keys:
        if key != higgs:
            hist, bins = np.histogram(
                field_clipped[detailed_labels_clipped == key],
                bins=bins,
                weights=weights_clipped[detailed_labels_clipped == key],
            )
            plt.stairs(hist_b, bins, fill=True, label=f"{key} bkg")
            hist_b -= hist
        else:
            print(key, hist_s.shape)

    plt.stairs(
        hist_s * mu_hat + hist_bkg,
        bins,
        fill=False,
        color="orange",
        label = f"$H \\rightarrow \\tau \\tau (\\mu = {mu_hat:.3f})$"
    )

    plt.stairs(
        hist_s + hist_bkg,
        bins,
        fill=False,
        color="red",
        label=f"$H \\rightarrow \\tau \\tau (\\mu = {1.0:.3f})$",
    )

    plt.legend()
    plt.title(f"Stacked histogram of {field_name}")
    plt.xlabel(f"{field_name}")
    plt.ylabel("Weighted count")
    plt.yscale(y_scale)
    plt.show()

def pair_plots_syst(dfall, df_syst, sample_size=100,columns=None):
    """
    Plots pair plots between the dataset and a system dataset.

    Args:
        * df_syst (DataFrame): The system dataset.
        * sample_size (int): The number of samples to consider (default: 10).
    
    ..images:: ../images/pair_plot_syst.png
    """
    
    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")
    
    df_sample = dfall[columns].copy()
    df_sample_syst = df_syst[columns].copy()

    index = np.random.choice(df_sample.index, sample_size, replace=False)
    df_sample = df_sample.loc[index]
    df_sample_syst = df_sample_syst.loc[index]
    df_sample["syst"] = False
    df_sample_syst["syst"] = True

    frames = [df_sample, df_sample_syst]
    del df_sample
    df_sample = pd.concat(frames)

    sns.set_theme(rc={"figure.figsize": (16, 14)}, style="whitegrid")

    ax = sns.PairGrid(df_sample, hue="syst")
    ax.map_upper(sns.scatterplot, alpha=0.5, size=0.3)
    ax.map_lower(
        sns.kdeplot, fill=True, levels=5, alpha=0.5
    )  # Change alpha value here
    ax.map_diag(
        sns.histplot,
        alpha=0.5,
        bins=20,
    )  # Change alpha value here
    ax.add_legend(title="Legend", labels=["syst", "no_syst"], fontsize=12)

    ax.figure.suptitle("Pair plots of features between syst and no_syst")
    plt.show()
    plt.close()

def histogram_syst(dfall, df_syst, weights, weight_syst, columns=None,nbin = 25):
    
    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")

    df_sample = dfall[columns].copy()
    df_sample_syst = df_syst[columns].copy()
            
    sns.set_theme(style="whitegrid")
    
    # Number of rows and columns in the subplot grid
    n_cols = 3  # Number of columns in the subplot grid
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

    for i, column in enumerate(columns):

        lower_percentile = 0
        upper_percentile = 97.5
        
        lower_bound = np.percentile(df_sample[column], lower_percentile)
        upper_bound = np.percentile(df_sample[column], upper_percentile)
        
        df_clipped = df_sample[(df_sample[column] >= lower_bound) & (df_sample[column] <= upper_bound)]
        weights_clipped = weights[(df_sample[column] >= lower_bound) & (df_sample[column] <= upper_bound)]
        
        df_clipped_syst = df_sample_syst[(df_sample_syst[column] >= lower_bound) & (df_sample_syst[column] <= upper_bound)] 
        weights_clipped_syst = weight_syst[(df_sample_syst[column] >= lower_bound) & (df_sample_syst[column] <= upper_bound)]
        
        min_value = df_clipped[column].min()
        max_value = df_clipped[column].max()

        # Define the bin edges
        bin_edges = np.linspace(min_value, max_value, nbin + 1)
        
        norminal_field = df_clipped[column]
        syst_field = df_clipped_syst[column]

        
        # Plot the histogram for label == 1 (Signal)
        axes[i].hist(norminal_field, bins=bin_edges, alpha=0.4, color='blue', label='Nominal', weights=weights_clipped, density=True)
        
        axes[i].hist(syst_field, bins=bin_edges, alpha=0.4, color='red', label='Systematics shifted', weights=weights_clipped_syst, density=True)    


        
        # Set titles and labels
        axes[i].set_title(f'{column}', fontsize=16)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Density')
        
        # Add a legend to each subplot
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
def event_vise_syst(dfall, df_syst, columns=None, sample_size=100):
    """
    Plots the event-wise shift between the nominal dataset and the systemalically shifted dataset.
    Args:
        * df_syst (DataFrame): The system dataset.
        * sample_size (int): The number of samples to consider (default: 100).
        * columns (list): The list of column names to consider (default: None, which includes all columns).
    
    ..Images:: ../images/event_vise_syst.png       
    """ 
    
    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")
    
    df_sample = dfall[columns].copy()
    df_sample_syst = df_syst[columns].copy()
    
    index = np.random.choice(df_sample.index, sample_size, replace=False)
    df_sample = df_sample.loc[index]
    df_sample_syst = df_sample_syst.loc[index]
    df_sample["syst"] = False
    df_sample_syst["syst"] = True
            
    sns.set_theme(style="whitegrid")
    
    # Number of rows and columns in the subplot grid
    n_cols = 3  # Number of columns in the subplot grid
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

    for i, column in enumerate(columns):
        field = df_sample[column]
        delta_field = df_sample_syst[column]-df_sample[column]
        axes[i].plot(field,delta_field, 'o', color='blue', label='No Syst')
        axes[i].set_title(f'{column}', fontsize=16)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('no_syst - syst')
        
        # Add a legend to each subplot
        axes[i].legend()


    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

        
def visualize_scatter(ingestion_result_dict, ground_truth_mus):
    """
    Plots a scatter Plot of ground truth vs. predicted mu values.

    Args:
        * ingestion_result_dict (dict): A dictionary containing the ingestion results.
        * ground_truth_mus (dict): A dictionary of ground truth mu values.
        
    .. Image:: images/scatter_plot_mu.png
    """
    plt.figure(figsize=(6, 4))
    for key in ingestion_result_dict.keys():
        ingestion_result = ingestion_result_dict[key]
        mu_hat = np.mean(ingestion_result["mu_hats"])
        mu = ground_truth_mus[key]
        plt.scatter(mu, mu_hat, c='b', marker='o')
    
    plt.xlabel('Ground Truth $\\mu$')
    plt.ylabel('Predicted $\\mu$ (averaged for 100 test sets)')
    plt.title('Ground Truth vs. Predicted $\\mu$ Values')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def visualize_coverage(ingestion_result_dict, ground_truth_mus):
    """
    Plots a coverage plot of the mu values.

    Args:
        * ingestion_result_dict (dict): A dictionary containing the ingestion results.
        * ground_truth_mus (dict): A dictionary of ground truth mu values.
        
    .. Image:: images/coverage_plot.png
    """

    for key in ingestion_result_dict.keys():
        plt.figure( figsize=(5, 5))

        ingestion_result = ingestion_result_dict[key]
        mu = ground_truth_mus[key]
        mu_hats = np.mean(ingestion_result["mu_hats"])
        p16s = ingestion_result["p16"]
        p84s = ingestion_result["p84"]
        
        # plot horizontal lines from p16 to p84
        for i, (p16, p84) in enumerate(zip(p16s, p84s)):
            if i == 0:
                plt.hlines(y=i, xmin=p16, xmax=p84, colors='b', label='Coverage interval')
            else:   
                plt.hlines(y=i, xmin=p16, xmax=p84, colors='b')

        plt.vlines(x=mu_hats, ymin=0, ymax=len(p16s), colors='r', linestyles='dashed', label='Predicted $\\mu$')
        plt.vlines(x=mu, ymin=0, ymax=len(p16s), colors='g', linestyles='dashed', label='Ground Truth $\\mu$')
        plt.xlabel("$\\mu$")
        plt.ylabel('pseudo-experiments')
        plt.title(f'$\\mu$ distribution - Set_{key}')
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
    plt.show()
