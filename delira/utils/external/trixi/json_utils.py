import json
import numpy as np
import traceback
from types import FunctionType, ModuleType
import warnings
import ast
import importlib

try:
    import torch
except ImportError as e:

    class torch:
        dtype = None


class CustomJSONEncoder(json.JSONEncoder):

    def _encode(self, obj):
        raise NotImplementedError

    def _encode_switch(self, obj):
        if isinstance(obj, list):
            return [self._encode_switch(item) for item in obj]
        elif isinstance(obj, dict):
            return {self._encode_key(key): self._encode_switch(val) for key, val in obj.items()}
        else:
            return self._encode(obj)

    def _encode_key(self, obj):
        return self._encode(obj)

    def encode(self, obj):
        return super(CustomJSONEncoder, self).encode(self._encode_switch(obj))

    def iterencode(self, obj, *args, **kwargs):
        return super(CustomJSONEncoder, self).iterencode(self._encode_switch(obj), *args, **kwargs)


class MultiTypeEncoder(CustomJSONEncoder):

    def _encode_key(self, obj):
        if isinstance(obj, int):
            return "__int__({})".format(obj)
        elif isinstance(obj, float):
            return "__float__({})".format(obj)
        else:
            return self._encode(obj)

    def _encode(self, obj):
        if isinstance(obj, tuple):
            return "__tuple__({})".format(obj)
        elif isinstance(obj, np.integer):
            return "__int__({})".format(obj)
        elif isinstance(obj, np.floating):
            return "__float__({})".format(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


class ModuleMultiTypeEncoder(MultiTypeEncoder):

    def _encode(self, obj, strict=False):
        if type(obj) == type:
            return "__type__({}.{})".format(obj.__module__, obj.__name__)
        elif type(obj) == torch.dtype:
            return "__type__({})".format(str(obj))
        elif isinstance(obj, FunctionType):
            return "__function__({}.{})".format(obj.__module__, obj.__name__)
        elif isinstance(obj, ModuleType):
            return "__module__({})".format(obj.__name__)
        else:
            try:
                return super(ModuleMultiTypeEncoder, self)._encode(obj)
            except Exception as e:
                if strict:
                    raise e
                else:
                    message = "Could not pickle object of type {}\n".format(
                        type(obj))
                    message += traceback.format_exc()
                    warnings.warn(message)
                    return repr(obj)


class CustomJSONDecoder(json.JSONDecoder):

    def _decode(self, obj):
        raise NotImplementedError

    def _decode_switch(self, obj):
        if isinstance(obj, list):
            return [self._decode_switch(item) for item in obj]
        elif isinstance(obj, dict):
            return {self._decode_key(key): self._decode_switch(val) for key, val in obj.items()}
        else:
            return self._decode(obj)

    def _decode_key(self, obj):
        return self._decode(obj)

    def decode(self, obj):
        return self._decode_switch(super(CustomJSONDecoder, self).decode(obj))


class MultiTypeDecoder(CustomJSONDecoder):

    def _decode(self, obj):
        if isinstance(obj, str):
            if obj.startswith("__int__"):
                return int(obj[8:-1])
            elif obj.startswith("__float__"):
                return float(obj[10:-1])
            elif obj.startswith("__tuple__"):
                return tuple(ast.literal_eval(obj[10:-1]))
        return obj


class ModuleMultiTypeDecoder(MultiTypeDecoder):

    def _decode(self, obj):
        if isinstance(obj, str):
            if obj.startswith("__type__"):
                str_ = obj[9:-1]
                module_ = ".".join(str_.split(".")[:-1])
                name_ = str_.split(".")[-1]
                type_ = str_
                try:
                    type_ = getattr(importlib.import_module(module_), name_)
                except Exception as e:
                    warnings.warn("Could not load {}".format(str_))
                return type_
            elif obj.startswith("__function__"):
                str_ = obj[13:-1]
                module_ = ".".join(str_.split(".")[:-1])
                name_ = str_.split(".")[-1]
                type_ = str_
                try:
                    type_ = getattr(importlib.import_module(module_), name_)
                except Exception as e:
                    warnings.warn("Could not load {}".format(str_))
                return type_
            elif obj.startswith("__module__"):
                str_ = obj[11:-1]
                type_ = str_
                try:
                    type_ = importlib.import_module(str_)
                except Exception as e:
                    warnings.warn("Could not load {}".format(str_))
                return type_
        return super(ModuleMultiTypeDecoder, self)._decode(obj)
