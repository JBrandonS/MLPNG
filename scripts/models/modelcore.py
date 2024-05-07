import argparse
import logging

from tensorflow.keras import Model
from scripts import Core
from scripts.utils.tf.dataloaders import PatchLoader

logger = logging.getLogger(__name__)

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
        raise ValueError(f"Model {name} not found")


def AutoModel(args=None, default="isensee"):
    """
    Creates an instance of the specified model class.

    This function parses command line arguments to determine the model class to instantiate.
    It then attempts to create an instance of the specified model class and returns it.
    If an error occurs during instantiation, it prints the error and returns None.

    Args:
        args (list, optional): A list of command line arguments. If not provided,
                               the function will use sys.argv by default.
                               The "--model" argument specifies the model class to instantiate.
                               The default model class is "isensee".

    Returns:
        model (object): An instance of the specified model class, or None if an error occurred.

    Raises:
        ValueError: If the specified model class cannot be found or instantiated.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=default, help="Model to use")
    pargs, _ = parser.parse_known_args(args)

    model_cls = pargs.model
    try:
        model = get_model_class(model_cls)(args)
        return model
    except ValueError as e:
        print(f"An error occurred: {e}")
        return None


class ModelCore(Core):

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
        *args: Variable length argument list to pass to the datasetLoader.
        **kwargs: Arbitrary keyword arguments to forward to the datasetLoader.

        Returns:
        The initialized dataset.
        """
        self._dataset = PatchLoader(self.patch_file, *args, **kwargs)
        return self._dataset

    def make_model(self, name=None, **kwargs):
        if self._keras_model is not None:
            raise ValueError("Model already created.")

        if self._dataset is None:
            logger.debug(
                "Dataset not initialized, initializing with default settings. "
            )
            self.init_dataset()

        # setup the name for the model, can have information such as slurm id
        self.name = name if name is not None else self.__class__.__name__

        inputs = self._dataset.input()
        outputs = self._model(inputs, **kwargs)
        self._keras_model = Model(inputs, outputs, name=self.name)

    def _model(self, inputs, *args, **kwargs):
        raise NotImplementedError(
            "_model not implemented. Please subclass ModelCore and implement _model."
        )

    ## override some functions from Core that need KSW which does not work with the models (TF)
    def _init_cosmo(self):
        pass

    def _init_almgen(self):
        pass

    #
    # I wanted a cleaner drop in for the keras.Model class, subclassing was bad
    # So we just pass through the calls to the keras.Model class
    #

    def _check_model(self):
        if self._keras_model is None or self._dataset is None:
            raise ValueError(
                "Model not fully initialized, please call init_dataset() and make_model() first. "
            )

    def compile(self, *args, **kwargs):
        self._check_model()
        return self._keras_model.compile(*args, **kwargs)

    def summary(self, *args, **kwargs):
        self._check_model()
        return self._keras_model.summary(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self._check_model()
        return self._keras_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self._check_model()
        return self._keras_model.predict(*args, **kwargs)

    def keras_model(self):
        self._check_model()
        return self._keras_model
