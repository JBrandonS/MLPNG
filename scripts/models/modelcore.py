import sys
import argparse
import logging

from tensorflow.keras import Model, Input

from scripts import Core
from scripts.utils.tf.dataloaders import PatchLoader

logger = logging.getLogger(__name__)


def _get(settings, name, default=None):
    """
    Get the value of a setting, providing the default if the setting is not found in settings.
    Logs information about the setting value if it is found and differs from the default, at DEBUG level.

    Args:
        name (str): The name of the setting.
        default (Any, optional): The default value to return if the setting is not found. Defaults to None.

    Returns:
        Any: The value of the setting if found, otherwise the default value.
    """
    val = settings.get(name, None)
    if val is None:
        logger.debug("Setting '%s' not found, using default: %s", name, repr(default))
        return default
    else:
        if val != default:
            logger.debug(
                "Found non-default value for '%s': %s (default: %s)",
                name,
                repr(val),
                repr(default),
            )
        return val


_MODEL_REG = {}


def register_model(cls):
    """Class decorator to register a new model."""

    logger.debug("Registering Model: %s", cls.__name__.upper())
    _MODEL_REG[cls.__name__.upper()] = cls
    return cls


def get_model_class(name):
    """Returns the model class given a name to allow for easy loading"""

    name = name.upper()  # make everything capitalized
    if name in _MODEL_REG:
        return _MODEL_REG[name]
    else:
        raise ValueError(f"Model {name} not found. ")


def AutoModel(args=None, default="isensee"):
    """
    Creates an instance of the specified model class.

    This function parses command line arguments to determine the model class to instantiate.
    It then attempts to create an instance of the specified model class and returns it.

    Args:
        args (list, optional): A list of command line arguments. If not provided,
                               the function will use sys.argv by default.
                               The "--model" argument specifies the model class to instantiate.
                               The default model class is "isensee".

    Returns:
        model (object): An instance of the specified model class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        help="Print the model summary",
    )
    parser.add_argument(
        "--tb", action=argparse.BooleanOptionalAction, help="Enable tensorboard logging"
    )
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, help="Enable wandb logging"
    )

    if args is None:
        args = sys.argv[1:]

    logger.debug("Parsing training CLI args: %s", args)
    pargs, args = parser.parse_known_args(args)

    # get the model and add settings
    model_cls = _get(vars(pargs), "model", default)
    model = get_model_class(model_cls)(args)
    model.print_summary = _get(vars(pargs), "summary", True)
    model.use_tensorboard = _get(vars(pargs), "tb", False)
    model.use_wandb = _get(vars(pargs), "wandb", False)
    return model


class ModelCore(Core):
    """
    Base class for model cores.

    Attributes:
    - _keras_model: The keras model object, set by make_model.
    - _dataset: The dataloader object, set by init_dataset.
    - BATCH_SIZE: The batch size for training, based on the nside of the data.
    """

    def __init__(self, argv=None):
        super().__init__(argv)

        self._keras_model = None  # the keras model object, set by make_model
        self._dataset = None  # the dataloader object, set by init_dataset

        # we want to use different batch sizes based on the nside due to memory
        # these may not work for all models so some will need to override BATCH_SIZE
        # but it seems to be a fairly sain defaults
        if self.nside <= 128:
            self.BATCH_SIZE = 128
        elif self.nside <= 256:
            self.BATCH_SIZE = 64
        elif self.nside <= 512:
            self.BATCH_SIZE = 32
        elif self.nside <= 1024:
            self.BATCH_SIZE = 16
        else:
            self.BATCH_SIZE = 8

    def init_dataset(self, *args, **kwargs):
        """
        Initializes the dataset for the model.

        Parameters:
        - *args: Variable length argument list to pass to the datasetLoader.
        - **kwargs: Arbitrary keyword arguments to forward to the datasetLoader.

        Returns:
        The initialized dataset.
        """
        self._dataset = PatchLoader(self.file_complete, *args, **kwargs)
        return self._dataset

    def make_model(self, name=None, **kwargs):
        """
        Creates the keras model.

        Parameters:
        - name: The name of the model.
        - **kwargs: Additional keyword arguments to pass to the model.

        Returns:
        None
        """

        if self._dataset is None:
            logger.debug(
                "Dataset not initialized, initializing with default settings. "
            )
            self.init_dataset()

        # setup the name for the model, can have information such as slurm id
        self.name = name if name is not None else self.name

        inputs = Input(self._dataset.shape)
        outputs = self._model(inputs, **kwargs)
        self._keras_model = Model(inputs, outputs, name=self.name)

    def _model(self, inputs, *args, **kwargs):
        """
        Abstract method to define the model architecture.

        Parameters:
        - inputs: The input tensor(s) to the model.
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.

        Returns:
        None
        """
        raise NotImplementedError(
            "_model not implemented. Please subclass ModelCore and implement _model."
        )

    #
    # I wanted a cleaner drop in for the keras.Model class, subclassing was bad
    # So we just pass through the calls to the keras.Model class
    #

    def _check_model(self):
        """
        Checks if the model and dataset are fully initialized.

        Raises:
        - ValueError: If the model or dataset is not fully initialized.

        Returns:
        None
        """
        if self._keras_model is None or self._dataset is None:
            raise ValueError(
                "Model not fully initialized, please call init_dataset() and make_model() first. "
            )

    def compile(self, *args, **kwargs):
        """
        Compiles the keras model.
        """
        self._check_model()
        return self._keras_model.compile(*args, **kwargs)

    def summary(self, *args, **kwargs):
        """
        Prints a summary of the keras model.
        """
        self._check_model()
        return self._keras_model.summary(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Trains the keras model.

        Parameters:
        - *args: Variable length argument list passed into the fit command.
        - **kwargs: Arbitrary keyword arguments passed into the fit command.

        Returns:
        The training history.
        """
        self._check_model()
        return self._keras_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Generates predictions using the keras model.

        Parameters:
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.

        Returns:
        The predicted values.
        """
        self._check_model()
        return self._keras_model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the model using the keras model.

        Parameters:
        - *args: Variable length argument list.
        - **kwargs: Arbitrary keyword arguments.

        Returns:
        The evaluation results.
        """
        self._check_model()
        return self._keras_model.evaluate(*args, **kwargs)

    def keras_model(self):
        """
        Returns the keras model.
        """
        self._check_model()
        return self._keras_model
