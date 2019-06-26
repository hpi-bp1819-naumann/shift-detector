from abc import ABCMeta, abstractmethod


class Preprocessor(metaclass=ABCMeta):

    @abstractmethod
    def __eq__(self, other):
        """Overrides the default implementation"""
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        """Overrides the default implementation"""
        raise NotImplementedError

    @abstractmethod
    def process(self, store):
        raise NotImplementedError
