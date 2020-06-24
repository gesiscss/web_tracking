from abc import ABC, abstractmethod
from ..utils.validator import Validator

class Measurement(ABC, Validator):
    """Abstract class for measurement model implementations."""

    @abstractmethod
    def calculate(self, df, **kwargs):
        pass
