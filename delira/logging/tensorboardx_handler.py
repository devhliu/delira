from collections.abc import Iterable
from tensorboardX import TorchVis
from logging import Handler, NOTSET
import logging

file_logger = logging.getLogger(__name__)


class TensorboardXHandler(Handler):
    """
    Handler for python's internal logging module to enable logging via 
    tensorboardX in both, visdom and tensorboard, by simply switching the 
    ``mode``

    Currently supported logging modes are 

    'scalar' | 'image' | 'figure' | 'histogram' | 'audio' | 'text' | 'graph' | 
    'onnx_graph' | 'embedding' | 'pr_curve' | 'video'

    Note
    ----
    Some of the backends may not support single logging modes. Have a look at 
    ` the github repository <https://github.com/lanpa/tensorboardX>`_ for 
    details

    """

    SUPPORTED_MODES = ["scalar", "image", "figure", "histogram", "audio",
                       "text", "graph", "onnx_graph", "embedding",
                       "pr_curve", "video"]

    def __init__(self, mode, level=NOTSET, logging_frequencies=None,
                 **init_kwargs):
        """

        Parameters
        ----------
        mode : str
            whether to use tensorboard or visdom backend.
            supported are 'tensorboard'|'visdom'
        level :  optional
            the handlers logging level (the default is NOTSET)
        logging_frequencies : None or int or Iterable or dict, optional
            the logging frequencies (can be specified per mode or for all 
            modes together).

            * If None : all frequencies are per default set to 1 
                (logging each iteration)

            * If int : Use this integer globally for all logging modes

            * If Iterable : Use the frequencies in the iterables's order. The 
                modes' order can be seen in the classes docstring 
                (missing values will be filled with 1s, left over values will 
                be ignored)

            * If dict : Use the frequencies for each mode based on key-mapping 
                (missing values will be filled with 1s, left over values will be ignored)

        **init_kwargs : 
            additional keyword arguments needed to specify the behaviour of 
            tensorboardX loggers

        See Also
        --------

        * `Visdom Writer <https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/visdom_writer.py#L24>`_

        * `Tensorflow Writer <https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/writer.py#L154>`_

        """

        super().__init__(level)

        self._logger = TorchVis(mode, **init_kwargs)

        self._frequencies = self._create_frequencies(logging_frequencies)
        self._counter = {k: 0 for k in self.SUPPORTED_MODES}

    def _create_frequencies(self, logging_frequencies):
        """
        Create the internal logging frequencies from the ``logging_frequencies``
        parameter

        Parameters
        ----------
        logging_frequencies : None or int or Iterable or dict, optional
            the logging frequencies (can be specified per mode or for all 
            modes together).

            * If None : all frequencies are per default set to 1 
                (logging each iteration)

            * If int : Use this integer globally for all logging modes

            * If Iterable : Use the frequencies in the iterables's order. The 
                modes' order can be seen in the classes docstring 
                (missing values will be filled with 1s, left over values will 
                be ignored)

            * If dict : Use the frequencies for each mode based on key-mapping 
                (missing values will be filled with 1s, left over values will be ignored)

        Raises
        ------
        ValueError
            If ``logging_frequencies are none of [None, int, Iterable, dict]

        Returns
        -------
        dict
            dictionary containing the internal logging frequencies for all 
            logging modes

        """

        if logging_frequencies is None:
            return {k: 1 for k in self.SUPPORTED_MODES}

        # set frequency for all modes to given int
        if isinstance(logging_frequencies, int):
            return {
                k: logging_frequencies for k in self.SUPPORTED_MODES}

        # set frequency for modes to given ints in order
        if isinstance(logging_frequencies, Iterable):
            if not len(logging_frequencies) == len(self.SUPPORTED_MODES):
                logging.warn("The number of logging frequencies and \
                    Supported Modes does not match, filling the other modes \
                    with frequency 1 or truncating them")

                missing_freqs = len(self.SUPPORTED_MODES) - \
                    len(logging_frequencies)

                if missing_freqs > 0:
                    logging_frequencies = logging_frequencies + \
                        [1] * missing_freqs

            return {k: _freq for k, _freq in zip(
                self.SUPPORTED_MODES, logging_frequencies)}

        # set frequency of given modes to value from dict -> association by key
        elif isinstance(logging_frequencies, dict):
            if not all(
                    [k in logging_frequencies for k in self.SUPPORTED_MODES]):
                logging.warn(
                    "Not all Supported Modes have a valid logging frequency, \
                        filling the other ones with 1")
            _frequencies = {}
            for key in self.SUPPORTED_MODES:
                _frequencies[key] = logging_frequencies.get(key, 1)

            return _frequencies

        else:
            raise ValueError("No valid type of logging frequencies found")

    def emit(self, record):
        """
        Processes the actual logged values

        Parameters
        ----------
        record : 
            the logged record; 
            For this handler the record message (``record.msg``) must be a 
            dict with the keys corresponding to the logging modes and the 
            value being a dict of keyword arguments

        """

        if not isinstance(record.msg, dict):
            # nothing to log here
            return

        for key, val in record.msg.items():
            if key in self.SUPPORTED_MODES:
                self._counter[key] += 1

                if (self._counter[key] % self._frequencies[key]) == 0:

                    getattr(self._logger, "add_" + key)(**val)

                else:
                    continue
