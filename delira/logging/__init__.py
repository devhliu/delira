from .multistream_handler import MultiStreamHandler
from .tensorboardx_handler import TensorboardXHandler
from delira import get_backends as __get_backends

if "TRIXI" in __get_backends():
    from .trixi_handler import TrixiHandler
