import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .utils import (
    setup_logging,
    save_data,
    remove_mono_dipole,
    print_errors,
    trim_alms,
    try_init_wandb,
    get_ksw_from_data,
    get_ksw_save_data,
)

from .plots import (
    plot_cl,
    plot_cl_alm,
    plot_cl_map,
    plot_histogram,
    plot_predictions,
    plot_predictions_combined,
    plot_mollview,
    plot_elsner_comp,
    plot_metrics,
    make_alm_plots,
    make_trainer_plots,
    plot_cl_vs,
)

from .slurm import Slurm

from .callbacks import RMSELoss, RMSEMetric, rmse_metrics

__all__ = [
    "setup_logging",
    "save_data",
    "remove_mono_dipole",
    "print_errors",
    "trim_alms",
    "try_init_wandb",
    "plot_cl",
    "plot_cl_alm",
    "plot_cl_map",
    "plot_histogram",
    "plot_predictions",
    "plot_predictions_combined",
    "plot_mollview",
    "plot_elsner_comp",
    "plot_metrics",
    "make_alm_plots",
    "Slurm",
    "RMSELoss",
    "RMSEMetric",
    "rmse_metrics",
    "make_trainer_plots",
    "get_ksw_from_data",
    "get_ksw_save_data",
    "plot_cl_vs",
]
