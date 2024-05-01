import logging

logger = logging.getLogger(__name__)


def try_init_wandb(notes=None, config=None):
    try:
        import wandb
        from wandb.keras import WandbMetricsLogger

        # wandb.tensorboard.patch(root_logdir=core.tb_dir)

        wandb.init(
            project="mlpng",
            notes=notes,
            tags=["dev"],
            config=config,
            dir="data",
            sync_tensorboard=True,
        )

        # Add the wandb logger to the callbacks, so it is used
        return WandbMetricsLogger()
    except ImportError:
        logger.error(
            "Wandb is not installed! "
            "Please install with `pip install wandb`. "
            "See: https://docs.wandb.ai/quickstart"
        )
        return
