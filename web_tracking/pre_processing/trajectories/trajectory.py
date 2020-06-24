from abc import ABC, abstractmethod

from ...utils.validator import Validator


class Trajectory(ABC, Validator):
    """Abstract class for the trajectory implementations."""

    def __init__(self, raw):
        self.raw = raw

    @abstractmethod
    def create(self, **kwargs):
        pass

    @abstractmethod
    def validate_input(self):
        pass
