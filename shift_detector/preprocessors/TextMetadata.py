import pandas as pd

from shift_detector.preprocessors.Preprocessor import Preprocessor
from shift_detector.utils.Miscellaneous import print_progress_bar
from shift_detector.utils.TextMetadataFunctions import md_functions

CATEGORICAL_METADATA_TYPES = frozenset(['unicode_categories', 'unicode_blocks', 'delimiter_type', 'languages'])
NUMERICAL_METADATA_TYPES = frozenset(['num_chars', 'ratio_upper','num_words', 'distinct_words', 'unique_words',
                                      'unknown_ratio', 'stopword_ratio', 'num_parts', 'complexity'])


class TextMetadata(Preprocessor):

    def __init__(self, text_metadata_types=None):
        if text_metadata_types is None:
            self.text_metadata_types = frozenset(['num_chars', 'num_words', 'distinct_words'])
        else:
            self.text_metadata_types = frozenset(text_metadata_types)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.text_metadata_types == other.text_metadata_types

    def __hash__(self):
        return hash(self.text_metadata_types)

    def selected_categorical_types(self):
        return self.text_metadata_types.intersection(CATEGORICAL_METADATA_TYPES)

    def selected_numerical_types(self):
        return self.text_metadata_types.intersection(NUMERICAL_METADATA_TYPES)

    def process(self, store):
        index = pd.MultiIndex.from_product([store.columns, self.text_metadata_types], names=['column', 'metadata'])
        metadata1 = pd.DataFrame(columns=index)
        metadata2 = pd.DataFrame(columns=index)
        for column in store.columns:
            clean1 = store.df1[column].dropna()
            clean2 = store.df2[column].dropna()
            print(column, ' - text metadata analysis:')
            for outer_iteration, metadata_type in enumerate(self.text_metadata_types):
                print('computing metadata ', metadata_type, ':')
                mdtype_values = []
                for inner_iteration, text in enumerate(clean1):
                    mdtype_values.append(md_functions(metadata_type)(text))
                    print_progress_bar(inner_iteration, len(clean1) - 1, 50)
                metadata1[(column, metadata_type)] = mdtype_values
                mdtype_values = []
                for inner_iteration, text in enumerate(clean2):
                    mdtype_values.append(md_functions(metadata_type)(text))
                    print_progress_bar(inner_iteration, len(clean2) - 1, 50)
                metadata2[(column, metadata_type)] = mdtype_values
        return metadata1, metadata2
