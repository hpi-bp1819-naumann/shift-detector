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

    def categorical_columns(self):
        return [column for column in self.text_metadata_types if column in CATEGORICAL_METADATA_TYPES]

    def numerical_columns(self):
        return [column for column in self.text_metadata_types if column in NUMERICAL_METADATA_TYPES]

    def process(self, first_df, second_df):
        metadata1 = pd.DataFrame()
        metadata2 = pd.DataFrame()
        for column in first_df.columns:
            clean1 = first_df[column].dropna()
            clean2 = second_df[column].dropna()
            print(column, ' - text metadata analysis:')
            for i, metadata_type in enumerate(self.text_metadata_types):
                print('computing metadata ', metadata_type, ':')
                mdtype_values = []
                for j, text in enumerate(clean1):
                    mdtype_values.append(md_functions(metadata_type)(text))
                    print_progress_bar(j, len(clean1) - 1, 50)
                metadata1[metadata_type] = mdtype_values
                # metadata1[metadata_type] = [md_functions(metadata_type)(text) for text in clean1]
                mdtype_values = []
                for j, text in enumerate(clean2):
                    mdtype_values.append(md_functions(metadata_type)(text))
                    print_progress_bar(j, len(clean2) - 1, 50)
                metadata2[metadata_type] = mdtype_values
                # metadata2[metadata_type] = [md_functions(metadata_type)(text) for text in clean2]
        return metadata1, metadata2
