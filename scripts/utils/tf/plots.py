import os
import matplotlib.pyplot as plt

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