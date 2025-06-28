# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from uniqa.dataclasses.document import Document
# from haystack.core.serialization import default_from_dict, default_to_dict
# from uniqa.dataclasses import ChatMessage, Document


@runtime_checkable
@dataclass
class Answer(Protocol):
    data: Any
    query: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:  # noqa: D102
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Answer":  # noqa: D102
        ...


@dataclass
class ExtractedAnswer:
    query: str
    score: float
    data: Optional[str] = None
    document: Optional[Document] = None
    context: Optional[str] = None
    document_offset: Optional["Span"] = None
    context_offset: Optional["Span"] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Span:
        start: int
        end: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.
        :returns:
            Deserialized object.
        """
        pass


@dataclass
class GeneratedAnswer:
    data: str
    query: str
    documents: List[Document]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.

        :returns:
            Deserialized object.
        """
        pass