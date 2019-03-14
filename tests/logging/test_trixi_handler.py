import os
import numpy as np
import logging
from delira import get_backends
import pytest


@pytest.mark.skipif("TRIXI" not in get_backends(), reason="trixi not installed")
def test_trixi_logger():
    from delira.logging import TrixiHandler
    from trixi.logger import NumpyPlotFileLogger

    handler = TrixiHandler(NumpyPlotFileLogger,
                           img_dir="./imgs", plot_dir="./plots")

    logging.basicConfig(level=logging.INFO,
                        handlers=[handler])

    logger = logging.getLogger(__name__)
    logger.info(
        {'image': {"image": np.random.rand(28, 28), "name": "test_img"}})


if __name__ == '__main__':
    test_trixi_logger()
