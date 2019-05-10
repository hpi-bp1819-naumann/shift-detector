from enum import Enum
import numpy as np
from gensim.models import FastText, Word2Vec

class EmbeddingType(Enum):
    FastText = "fasttext"
    Word2Vec = "word2vec"

class WordEmbedding():

    def __init__(self, embedding=None, model=None, trained_model=None):
        self.embedding = embedding
        self.model = model
        self.trained_model = trained_model
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__):
            if self.trained_model == other.trained_model \
                or self.model == other.model \
                or self.embedding == other.embedding:
                return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        if not self.embedding and not self.model:
            return hash(self.trained_model)
        if not self.embedding:
            return hash(self.model)
        return hash(self.embedding)
    
    def process(self, train_df, test_df):
        df = train_df.dropna().str.lower().str.split()

        if not self.trained_model:
            if not self.model:
                if self.embedding == EmbeddingType.FastText:
                    self.model = FastText(size=300, window=5, min_count=1, workers=4)
                elif self.embedding == EmbeddingType.Word2Vec:
                    self.model = Word2Vec(size=300, window=5, min_count=1, workers=4)
                else:
                    raise Exception('No embedding defined')

            self.model.build_vocab(sentences=df)
            # TODO: replace 'epochs = 10'
            self.model.train(sentences=df, total_examples=len(df), epochs=1)
            self.trained_model = self.model

        processed = df.apply(lambda row: np.sum([self.trained_model.wv[word] for word in row], axis=0))
        return processed
