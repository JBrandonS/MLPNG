import logging

logger = logging.getLogger(__name__)


def try_init_wandb(project="mlpng", notes=None, tags=[], config=None, dir="data", append_to=None):
    """
    Tries to initialize the wandb library for logging experiments.

    Args:
        notes (str, optional): A longer description of the run, like a -m commit message in git. This helps you remember what you were doing when you ran this run.
        config (dict, argparse, absl.flags, str, optional): This sets wandb.config, a dictionary-like object for saving inputs to your job, like hyperparameters for a model or settings for a data preprocessing job. The config will show up in a table in the UI that you can use to group, filter, and sort runs. Keys should not contain . in their names, and values should be under 10 MB. If dict, argparse or absl.flags: will load the key value pairs into the wandb.config object. If str: will look for a yaml file by that name, and load config from that file into the wandb.config object.
        append_to (list, optional): List to append the WandbMetricsLogger to. Defaults to None.

    Returns:
        WandbMetricsLogger: The WandbMetricsLogger object for logging metrics to Weights & Biases.
    """
    try:
        import wandb
        from wandb.keras import WandbMetricsLogger

        # wandb.tensorboard.patch(root_logdir=core.tb_dir)

        wandb.init(
            project=project,
            notes=notes,
            tags=["dev"] + tags,
            config=config,
            dir=dir,
        )

        # Add the wandb logger to the callbacks, so it is used
        if append_to:
            append_to.append(WandbMetricsLogger())

        return WandbMetricsLogger()
    except ImportError:
        logger.error(
            "Wandb is not installed! "
            "Please install with `pip install wandb`. "
            "See: https://docs.wandb.ai/quickstart"
        )
        return
