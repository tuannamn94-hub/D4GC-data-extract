"""Base class for content extractors."""

from __future__ import annotations

import abc
from typing import Any


class ExtractionResult:
    """The output of an extractor: text plus optional structure."""

    def __init__(self, text: str, structure: dict[str, Any] | None = None):
        self.text = text
        self.structure = structure or {}


class Extractor(abc.ABC):
    """A parser that turns raw file bytes into text + structure.

    Subclasses declare which file types they handle via ``file_types`` and
    implement :meth:`extract`.
    """

    #: iterable of canonical file-type names this extractor handles
    file_types: tuple[str, ...] = ()

    @abc.abstractmethod
    def extract(self, path: str) -> ExtractionResult:
        """Extract text and structure from the file at ``path``."""
        raise NotImplementedError
