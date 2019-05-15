import abc

class NGram:

    def __init__(self, n):
        self.n = n
        if self.n < 1:
            raise Exception('n has to be greater than 0')

    def __eq__(self, other):
        """Overrides the default implementation"""
        return self.n == other.n

    def __hash__(self):
        return hash((self.__class__, self.n))

    @abc.abstractmethod
    def process(self, train, test):
        return
