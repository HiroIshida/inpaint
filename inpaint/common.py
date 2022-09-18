import base64
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Type, TypeVar

import numpy as np

JsonableT = TypeVar("JsonableT", bound="Jsonable")


@dataclass
class Jsonable:
    def serialize(self) -> str:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                value_new = ["np.ndarray", base64.b64encode(pickle.dumps(value)).decode("utf-8")]
                d[key] = value_new
        return json.dumps(d)

    @classmethod
    def deserialize(cls: Type[JsonableT], string: str) -> JsonableT:
        d = json.loads(string)
        for key, value in d.items():
            if isinstance(value, list):
                if value[0] == "np.ndarray":
                    d[key] = pickle.loads(base64.b64decode(value[1].encode()))
        return cls(**d)


@dataclass
class RequestData(Jsonable):
    image: np.ndarray
    mask: np.ndarray
    use_gpu: bool = False
    metadata: Optional[Dict] = None

    def __post_init__(self):
        assert self.image.ndim == 3
        assert self.image.dtype == np.uint8
        assert self.mask.ndim == 2
        assert self.mask.dtype == np.uint8
        assert self.image.shape[:2] == self.mask.shape

        mask_element_set = set(self.mask.flatten().tolist())
        is_binary_arr = mask_element_set == {0, 255}
        assert is_binary_arr, mask_element_set

        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResponseData(Jsonable):
    image: np.ndarray
    debug_image: np.ndarray
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
