import logging

logger = logging.getLogger(__name__)


def try_init_wandb(
    project="mlpng",
    notes=None,
    tags=[],
    config=None,
    dir="data",
    append_to=None,
    patch_tb=False,
    patch_logdir=None,
    **kwargs
):
    """
    Initialize and configure the Weights & Biases (wandb) library for logging experiments.

    Parameters:
    - project (str): The name of the project to which the run belongs. Default is "mlpng".
    - notes (str): A description or notes for the run. Default is None.
    - tags (list): A list of tags to associate with the run. Default is an empty list.
    - config (dict): A dictionary of configuration parameters for the run. Default is None.
    - dir (str): The directory where the run files will be saved. Default is "data".
    - append_to (list): A list of callbacks to which the WandbMetricsLogger callback will be appended. Default is None.
    - patch_tb (bool): Whether to patch TensorBoard logging. Default is False.
    - patch_logdir (str): The root log directory for patching TensorBoard logging. Required if patch_tb is True.
    - **kwargs: Additional keyword arguments to pass to wandb.init().

    Returns:
    - WandbMetricsLogger: The WandbMetricsLogger callback object.

    Raises:
    - ValueError: If patch_tb is True but patch_logdir is not set.
    """
    try:
        import wandb
        from wandb.keras import WandbMetricsLogger
    except ImportError:
        logger.error(
            "Wandb is not installed! "
            "Please install with `pip install wandb`. "
            "See: https://docs.wandb.ai/quickstart"
        )
        return
    
    if patch_tb:
        if patch_logdir is None:
            raise ValueError("If patch_tb is True, patch_logdir must be set.")
        wandb.tensorboard.patch(root_logdir=patch_logdir)

    wandb.init(
        project=project,
        notes=notes,
        tags=tags,
        config=config,
        dir=dir,
        **kwargs
    )

    # Add the wandb logger to the callbacks, so it is used
    if append_to:
        append_to.append(WandbMetricsLogger())

    return WandbMetricsLogger()
