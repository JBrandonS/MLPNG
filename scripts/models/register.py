import logging
logger = logging.getLogger(__name__)

# global storage of model classes
_MODEL_REG = {}


def register_model(cls):
    """Class decorator to register a new model."""
    logger.debug("Registering Model: %s", cls.__name__.upper())
    _MODEL_REG[cls.__name__.upper()] = cls
    return cls


def get_model_class(name):
    """Returns the model class given a name to allow for easy loading

    Usage:
    ```python
        alm_class = get_model_class("alm")
        model = alm_class(core, dataset_class, ...)
    ```
    """

    name = name.upper()  # make everything capitalized
    if name in _MODEL_REG:
        return _MODEL_REG[name]
    else:
        raise ValueError(f"Model {name} not found")