class NGram:

    def __init__(self, n=5):
        self.n = n
        if self.n < 1:
            raise Exception('n has to be greater than 0')

    def __eq__(self, other):
        """Overrides the default implementation"""
        return self.n == other.n

    def __hash__(self):
        return hash((self.__class__, self.n))

    def process(self, train, test):

        def generate_ngram(text, n):
            ngram = {}
            for i in range(len(text) - n + 1):
                ngram[text[i:i+n]] = 1 if text[i:i+n] not in ngram else ngram[text[i:i+n]] + 1
            return ngram

        train = train.dropna().str.lower()
        train_processed = train.apply(lambda row: generate_ngram(row, self.n))

        test = test.dropna().str.lower()
        test_processed = test.apply(lambda row: generate_ngram(row, self.n))

        return train_processed, test_processed
