# serialization.py 是一个模块，用于序列化和反序列化对象。

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from unittest.mock import Mock
from typing import Any, Dict
import pytest

# from uniqa.core.pipeline import Pipeline
# from uniqa.core.component import component
from uniqa.core.errors import DeserializationError, SerializationError
# from uniqa.testing import factory
from uniqa.core.serialization import (
    default_to_dict,
    default_from_dict,
    generate_qualified_class_name,
    import_class_by_name,
    component_to_dict,
)


class MyClass(object):
    __name__ = 'MyClass'
    def __init__(self, my_param: int = 10):
        self.my_param = my_param

    def to_dict(self):
        return default_to_dict(
            self, 
            my_param=self.my_param
        )

    @classmethod
    def from_dict(self, data: Dict[str, Any]):
        """
        修改 from_dict 为类方法，接收的是类 MyClass 而不是实例
        """
        init_params = data["init_parameters"]
        return default_from_dict(self, data)


obj = MyClass(my_param=5)
data = obj.to_dict()
print(data)
assert data == {
    "type": "__main__.MyClass",
    "init_parameters": {
        "my_param": 5,
    },
}

comp = obj.from_dict(data)
print(comp.to_dict())
