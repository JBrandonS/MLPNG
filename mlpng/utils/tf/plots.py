import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

def plot_predictions(y_val, y_pred, save_file=None, fisher=None, scaled_variance=None):
    df = pd.DataFrame(
        {"True Labels": y_val.flatten(), "Predicted Labels": y_pred.flatten()}
    )

    # Create a scatter plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="True Labels", y="Predicted Labels")

    # Truth line
    plt.plot(
        [min(y_val), max(y_val)], [min(y_val), max(y_val)], color="red", linestyle="--"
    )

    if fisher is not None:
        std_dev = np.sqrt(1 / fisher)
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) + std_dev, max(y_val) + std_dev],
            color="blue",
            linestyle="--",
            label="Fisher",
        )
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) - std_dev, max(y_val) - std_dev],
            color="blue",
            linestyle="--",
        )

    if scaled_variance is not None:
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) + scaled_variance, max(y_val) + scaled_variance],
            color="green",
            linestyle="--",
            label=r"Scaled Variance (1/$\sqrt{f_{sky} f}$)",
        )
        plt.plot(
            [min(y_val), max(y_val)],
            [min(y_val) - scaled_variance, max(y_val) - scaled_variance],
            color="green",
            linestyle="--",
        )

    # Line for perfect fit
    r2 = r2_score(df["True Labels"], df["Predicted Labels"])
    plt.text(min(y_val), max(y_val), f"R^2 = {r2:.2f}", verticalalignment="top")

    plt.title("Predicted vs True Labels")
    if save_file is not None:
        plt.savefig(save_file)


def plot_histogram(y_val, y_pred, save_file=None):
    """Plot and save a histogram of predictions with mean and std dev as title"""
    # Calculate mean and standard deviation
    y_val = y_val.flatten()
    y_pred = y_pred.flatten()

    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)

    diff = y_pred - y_val
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, figsize=(12, 12))

    # Plot the predictions on the first subplot
    sns.histplot(y_pred, ax=axs[0], legend=False)
    axs[0].set_title(f"Predictions - Mean: {mean_pred:.2f}, Standard Deviation: {std_pred:.2f}")

    # Plot the differences on the second subplot
    sns.histplot(diff, ax=axs[1], legend=False)
    axs[1].set_title(f"Differences - Mean: {mean_diff:.2f}, Standard Deviation: {std_diff:.2f}")

    # Save the plot
    if save_file is not None:
        plt.savefig(save_file)


def plot_metrics(history, save_file=None, metrics=["loss"]):
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, figsize=(15, 6 * num_metrics))

    if num_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        axs[i].plot(history.history[metric])
        axs[i].plot(history.history[f"val_{metric}"])
        axs[i].set_title(f"Model {metric}")
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Epoch")
        axs[i].legend(["Train", "Validation"], loc="upper right")

    if save_file is not None:
        plt.savefig(save_file)

def plot_activations(model, ds, name, layers=None, plot_dir="data/plots/activations"):
    import keract  # pip install keract for this to work

    first_batch = next(iter(ds.take(1)))
    images, _ = first_batch
    img = images[0][None, :, :, :]  # need to add back in the batch dim

    activations = keract.get_activations(model, img, auto_compile=True)

    if layers is not None:
        activations = activations.get(layers)

    keract.display_activations(
        activations,
        save=True,
        directory=os.path.join(plot_dir, name),
        data_format="channels_last",
    )