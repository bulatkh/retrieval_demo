from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VLMWrapperRetrieval(ABC):
    model: Optional[Any] = field(default=None)
    processor: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.model is None or self.processor is None:
            raise ValueError("Both model and processor must be provided")

    @abstractmethod
    def process_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_embeddings(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_text_embeddings(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_image_embeddings(self, *args, **kwargs):
        pass


@dataclass
class VLMWrapperCaptioning(ABC):
    model: Optional[Any] = field(default=None)
    processor: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.model is None or self.processor is None:
            raise ValueError("Both model and processor must be provided")

    @abstractmethod
    def process_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass
