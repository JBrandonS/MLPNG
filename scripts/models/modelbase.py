import logging

from tensorflow.keras import Model
from scripts.utils.tf.dataloaders import PatchLoader

logger = logging.getLogger(__name__)


class ModelBase:

    def __init__(self, core, dataset_class=PatchLoader, dataset_file=None):
        self.core = core
        self._ds_class = dataset_class
        self._ds_file = dataset_file if dataset_file is not None else core.patch_file
        self.__model = None
        self._ds = None

        # we want to use different batch sizes based on the nside due to memory
        if core.nside <= 128:
            self.BATCH_SIZE = 128
        elif core.nside <= 256:
            self.BATCH_SIZE = 64
        elif core.nside <= 512:
            self.BATCH_SIZE = 32
        elif core.nside <= 1024:
            self.BATCH_SIZE = 16
        else:
            self.BATCH_SIZE = 8

    def _model(self, inputs, *args, **kwargs):
        raise NotImplementedError("_model not implemented")

    def dataset(self, *args, **kwargs):
        self._ds = self._ds_class(self._ds_file, *args, **kwargs)
        return self._ds

    def make_model(self, *args, **kwargs):
        if self._ds is None:
            raise ValueError("Dataset not set, please call dataset() first")

        if self.__model is not None:
            raise ValueError("Model already created.")

        name = kwargs.pop("name", "")
        inputs = self._ds.input()
        outputs = self._model(inputs, *args, **kwargs)
        self.__model = Model(inputs, outputs, name=name)

    # I wanted a cleaner drop in for the keras.Model class, subclassing was bad
    # So we just pass through the calls to the keras.Model class
    def _check_model(self):
        if self.__model is None:
            raise ValueError(
                "Model not fully initialized, please call dataset() and make_model() first"
            )

    def compile(self, *args, **kwargs):
        self._check_model()
        return self.__model.compile(*args, **kwargs)

    def summary(self, *args, **kwargs):
        self._check_model()
        return self.__model.summary(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self._check_model()
        return self.__model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self._check_model()
        return self.__model.predict(*args, **kwargs)
