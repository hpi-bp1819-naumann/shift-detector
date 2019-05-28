import logging as logger
import re
import unicodedata
from abc import abstractmethod

import pandas as pd
# noinspection PyPackageRequirements
from iso639 import languages
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
# noinspection PyPackageRequirements
from spellchecker import SpellChecker
from textstat import textstat

from shift_detector.preprocessors.Preprocessor import Preprocessor
from shift_detector.utils import UCBlist
from shift_detector.utils.ColumnManagement import ColumnType
from shift_detector.utils.TextMetadataUtils import dictionary_to_sorted_string, tokenize, delimiter_sentence, \
    delimiter_other, delimiter_HTML


class GenericTextMetadata(Preprocessor):

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

    @staticmethod
    @abstractmethod
    def metadata_name() -> str:
        pass

    @abstractmethod
    def metadata_column_type(self) -> ColumnType:
        pass

    @abstractmethod
    def metadata_function(self, text):
        pass

    def process(self, store):
        metadata1 = pd.DataFrame()
        metadata2 = pd.DataFrame()
        df1, df2 = store[ColumnType.text]
        for column in df1.columns:
            clean1 = df1[column].dropna()
            clean2 = df2[column].dropna()
            logger.info(self.metadata_name(), ' analysis for ', column, ':')
            metadata1[column] = [self.metadata_function(text) for text in clean1]
            metadata2[column] = [self.metadata_function(text) for text in clean2]
        return metadata1, metadata2


class NumCharsMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'num_chars'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        return len(text)


class RatioUpperMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'ratio_upper'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        if text == "":
            return 0
        lower = sum(map(str.islower, text))
        upper = sum(1 for c in text if c.isupper())
        return round((upper * 100) / (lower + upper), 2)


class UnicodeCategoriesMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'unicode_categories'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    @staticmethod
    def unicode_category_histogram(text):
        characters = {}
        for c in text:
            category = unicodedata.category(c)
            if category in characters:
                characters[category] += 1
            else:
                characters[category] = 1
        return characters

    def metadata_function(self, text):
        return dictionary_to_sorted_string(self.unicode_category_histogram(text))


class UnicodeBlocksMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'unicode_blocks'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    @staticmethod
    def unicode_block_histogram(text):

        def block(character):
            """ Return the Unicode block name for ch, or None if ch has no block.
            from https://stackoverflow.com/questions/243831/unicode-block-of-a-character-in-python
            :param character"""
            assert isinstance(character, str) and len(character) == 1, repr(character)
            cp = ord(character)
            for start, end, name in UCBlist._blocks:
                if start <= cp <= end:
                    return name

        characters = {}
        for c in text:
            category = block(c)
            if category in characters:
                characters[category] += 1
            else:
                characters[category] = 1
        return characters

    def metadata_function(self, text):
        return dictionary_to_sorted_string(self.unicode_block_histogram(text))


class NumWordsMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'num_words'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        return len(tokenize(text))


class NumDistinctWordsMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'distinct_words'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        distinct_words = []
        words = tokenize(text)
        for word in words:
            if word not in distinct_words:
                distinct_words.append(word)
        return len(distinct_words)


class NumUniqueWordsMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'unique_words'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        words = tokenize(text)
        seen_once = []
        seen_often = []
        for word in words:
            if word not in seen_often:
                if word not in seen_once:
                    seen_once.append(word)
                else:
                    seen_once.remove(word)
                    seen_often.append(word)
        return len(seen_once)


class UnknownWordRatioMetadata(GenericTextMetadata):

    def __init__(self, language='en'):
        self.language = language

    @staticmethod
    def metadata_name() -> str:
        return 'unknown_word_ratio'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        # not working for every language
        words = tokenize(text)
        spell = SpellChecker(self.language)

        if len(words) == 0:
            return 0.0

        misspelled = spell.unknown(words)
        return len(misspelled) / len(words)


class StopwordRatioMetadata(GenericTextMetadata):

    def __init__(self, language='en'):
        self.language = language

    @staticmethod
    def metadata_name() -> str:
        return 'stopword_ratio'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        # not working for every language
        stopword_count = 0
        words = tokenize(text)
        try:
            stop = stopwords.words(languages.get(part1=self.language).name.lower())
            if (len(words) == 0):
                return 0.0
            for word in words:
                if word.lower() in stop:
                    stopword_count += 1
            return stopword_count/ len(words)
        except OSError as error:
            raise ValueError('Language ' + languages.get(
                part1=self.language).name.lower() + ' is not supported for stopword ratio') from error


class DelimiterTypeMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'delimiter_type'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    def metadata_function(self, text):
        html = re.compile('<.*?>')
        point = re.compile(delimiter_sentence)
        other = re.compile(delimiter_other)
        if (html.search(text)):
            return 'html'
        elif (point.search(text)):
            return 'sentence'
        elif (other.search(text)):
            return 'other delimiter'
        elif (len(text) == 0):
            return 'empty'
        else:
            return 'no delimiter'


class NumPartsMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'num_parts'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        if DelimiterTypeMetadata().metadata_function(text) == 'html':
            return len(re.split(delimiter_HTML, text))
        elif DelimiterTypeMetadata().metadata_function(text) == 'sentence':
            return len(re.split(delimiter_sentence, text))
        elif DelimiterTypeMetadata().metadata_function(text) == 'other delimiter':
            return len(re.split(delimiter_other, text))
        else:
            return 0


DetectorFactory.seed = 0  # seed language detection to make it deterministic


class LanguageMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'language'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    @staticmethod
    def detect_languages(text):
        if len(text) == 0:
            detect(text)  # trigger LangDetectException. Throwing one in here smh doesnt work
        if DelimiterTypeMetadata().metadata_function(text) == 'html':
            parts = re.split(r'<\s*br\s*/?>', text)
        else:
            parts = re.split(r'[\n\r]+', text)
        parts = [x.strip() for x in parts if x.strip()]
        detected_languages = {}
        for part in parts:
            lang = detect(part)
            if lang in detected_languages:
                detected_languages[lang] += 1
            else:
                detected_languages[lang] = 1
        return detected_languages

    def metadata_function(self, text):
        return dictionary_to_sorted_string(self.detect_languages(text))


class ComplexityMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'complexity'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        # lower value means more complex
        # works best for longer english texts. kinda works for other languages as well (not good though)
        return textstat.flesch_reading_ease(text)


class TextMetadata(Preprocessor):

    def __init__(self, text_metadata_types=None):
        if text_metadata_types is None:
            self.text_metadata_types = frozenset([NumCharsMetadata(), NumWordsMetadata(), NumDistinctWordsMetadata()])
        else:
            self.text_metadata_types = frozenset(text_metadata_types)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.text_metadata_types == other.text_metadata_types

    def __hash__(self):
        return hash((self.__class__, self.text_metadata_types))

    def process(self, store):
        df1, df2 = store[ColumnType.text]
        metadata_names = sorted([mdtype.metadata_name() for mdtype in self.text_metadata_types])
        index = pd.MultiIndex.from_product([df1.columns, metadata_names], names=['column', 'metadata'])
        metadata1 = pd.DataFrame(columns=index)
        metadata2 = pd.DataFrame(columns=index)
        for metadata_type in self.text_metadata_types:
            md1, md2 = store[metadata_type]
            for column in df1.columns:
                metadata1[(column, metadata_type.metadata_name())] = md1[column]
                metadata2[(column, metadata_type.metadata_name())] = md2[column]
        return metadata1, metadata2