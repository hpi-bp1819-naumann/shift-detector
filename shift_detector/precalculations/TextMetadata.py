import logging as logger
import re
import regex
import unicodedata
from abc import abstractmethod
from collections import defaultdict

import pandas as pd
# noinspection PyPackageRequirements
from iso639 import languages
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
# noinspection PyPackageRequirements
from spellchecker import SpellChecker
from textstat import textstat

from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.utils import UCBlist
from shift_detector.utils.ColumnManagement import ColumnType
from shift_detector.utils.TextMetadataUtils import dictionary_to_sorted_string, delimiters


class GenericTextMetadata(Precalculation):

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
            logger.info(self.metadata_name() + ' analysis for ' + column)
            metadata1[column] = [self.metadata_function(text) for text in clean1]
            metadata2[column] = [self.metadata_function(text) for text in clean2]
        return metadata1, metadata2


class GenericTextMetadataWithTokenizing(GenericTextMetadata):

    @staticmethod
    @abstractmethod
    def metadata_name() -> str:
        pass

    @abstractmethod
    def metadata_column_type(self) -> ColumnType:
        pass

    @abstractmethod
    def metadata_function(self, words):
        pass

    def process(self, store):
        metadata1 = pd.DataFrame()
        metadata2 = pd.DataFrame()
        df1, df2 = store[TokenizeIntoWords()]
        for column in df1.columns:
            logger.info(self.metadata_name() + ' analysis for ' + column)
            metadata1[column] = [self.metadata_function(words) for words in df1[column]]
            metadata2[column] = [self.metadata_function(words) for words in df2[column]]
        return metadata1, metadata2


class GenericTextMetadataWithTokenizingAndLanguage(GenericTextMetadata):

    def __init__(self, language='en', infer_language=False):
        self.language = language
        self.infer_language = infer_language

    @staticmethod
    @abstractmethod
    def metadata_name() -> str:
        pass

    @abstractmethod
    def metadata_column_type(self) -> ColumnType:
        pass

    @abstractmethod
    def metadata_function(self, language_and_words):
        pass

    def process(self, store):
        metadata1 = pd.DataFrame()
        metadata2 = pd.DataFrame()
        df1, df2 = store[TokenizeIntoWords()]
        if self.infer_language:
            lang1, lang2 = store[LanguageMetadata()]
        for column in df1.columns:
            logger.info(self.metadata_name() + ' analysis for ' + column)
            temp_column1 = []
            temp_column2 = []
            for i in range(len(df1)):
                if self.infer_language:
                    temp_column1.append(self.metadata_function((lang1[column][i], df1[column][i])))
                    temp_column2.append(self.metadata_function((lang2[column][i], df2[column][i])))
                else:
                    temp_column1.append(self.metadata_function((self.language, df1[column][i])))
                    temp_column2.append(self.metadata_function((self.language, df2[column][i])))
            metadata1[column] = temp_column1
            metadata2[column] = temp_column2
        return metadata1, metadata2


class TokenizeIntoWords(Precalculation):

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

    def tokenize_into_words(self, text):
        text = re.sub(r"-", ' ', text)
        text = re.sub(r"[^\w\s']", '', text)
        splitted = re.split(r'\W\s|\s', text)
        while '' in splitted:
            splitted.remove('')
        return splitted

    def process(self, store):
        tokenized1 = pd.DataFrame()
        tokenized2 = pd.DataFrame()
        df1, df2 = store[ColumnType.text]
        for column in df1.columns:
            clean1 = df1[column].dropna()
            clean2 = df2[column].dropna()
            tokenized1[column] = [self.tokenize_into_words(text) for text in clean1]
            tokenized2[column] = [self.tokenize_into_words(text) for text in clean2]
        return tokenized1, tokenized2


class NumCharsMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'num_chars'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        return len(text)


class RatioUppercaseLettersMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'ratio_upper'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        if text == "":
            return 0
        alpha = sum(1 for c in text if c.isalpha())
        upper = sum(1 for c in text if c.isupper())
        return upper / alpha


class UnicodeCategoriesMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'unicode_categories'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    @staticmethod
    def unicode_category_histogram(text):
        characters = defaultdict(int)
        for c in text:
            category = unicodedata.category(c)
            characters[category] += 1
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
            """ Return the Unicode block name for character, or None if character has no block.
            from https://stackoverflow.com/questions/243831/unicode-block-of-a-character-in-python
            :param character"""
            assert isinstance(character, str) and len(character) == 1, repr(character)
            cp = ord(character)
            for start, end, name in UCBlist.blocks:
                if start <= cp <= end:
                    return name

        characters = defaultdict(int)
        for c in text:
            category = block(c)
            characters[category] += 1
        return characters

    def metadata_function(self, text):
        return dictionary_to_sorted_string(self.unicode_block_histogram(text))


class NumWordsMetadata(GenericTextMetadataWithTokenizing):

    @staticmethod
    def metadata_name() -> str:
        return 'num_words'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, words):
        return len(words)


class DistinctWordsRatioMetadata(GenericTextMetadataWithTokenizing):

    @staticmethod
    def metadata_name() -> str:
        return 'distinct_words'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, words):
        distinct_words = set()
        if len(words) == 0:
            return 0.0
        for word in words:
            if word not in distinct_words:
                distinct_words.add(word)
        return len(distinct_words) / len(words)


class UniqueWordsRatioMetadata(GenericTextMetadataWithTokenizing):

    @staticmethod
    def metadata_name() -> str:
        return 'unique_words'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, words):
        if len(words) == 0:
            return 0.0
        seen_once = set()
        seen_often = set()
        for word in words:
            if word not in seen_often:
                if word not in seen_once:
                    seen_once.add(word)
                else:
                    seen_once.remove(word)
                    seen_often.add(word)
        return len(seen_once) / len(words)


class UnknownWordRatioMetadata(GenericTextMetadataWithTokenizingAndLanguage):

    @staticmethod
    def metadata_name() -> str:
        return 'unknown_word_ratio'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, language_and_words):
        # pyspellchecker supports multiple languages including English, Spanish, German, French, and Portuguese
        language = language_and_words[0]
        words = language_and_words[1]
        try:
            spell = SpellChecker(language)
        except ValueError as error:
            raise ValueError('The language ' +
                             languages.get(part1=language).name.lower() +
                             ' is not supported by UnknownWordRatioMetadata') from error
        if len(words) == 0:
            return 0.0

        misspelled = spell.unknown(words)
        return len(misspelled) / len(words)


class StopwordRatioMetadata(GenericTextMetadataWithTokenizingAndLanguage):

    @staticmethod
    def metadata_name() -> str:
        return 'stopword_ratio'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, language_and_words):
        # not working for every language
        language = language_and_words[0]
        words = language_and_words[1]
        stopword_count = 0
        try:
            stop = stopwords.words(languages.get(part1=language).name.lower())
            if len(words) == 0:
                return 0.0
            for word in words:
                if word.lower() in stop:
                    stopword_count += 1
            return stopword_count / len(words)
        except OSError as error:
            raise ValueError('The language ' +
                             languages.get(part1=self.language).name.lower() +
                             ' is not supported by StopwordRatioMetadata') from error


class DelimiterTypeMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'delimiter_type'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    def metadata_function(self, text):
        for key, value in delimiters.items():
            if regex.compile(value).search(text):
                return key
        return 'no delimiter'


class NumPartsMetadata(GenericTextMetadata):
    # Calculates the delimiter of the text and then splits the text by its delimiter to calculate the number of parts in the text

    @staticmethod
    def metadata_name() -> str:
        return 'num_parts'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        delimiter = DelimiterTypeMetadata().metadata_function(text)
        for key, value in delimiters.items():
            if key == delimiter:
                return len(regex.split(regex.compile(value), text))
        return 0


class LanguagePerParagraph(GenericTextMetadata):
    # Depending on the texts delimiter splits the text into parts and calculates the language for each part. 
    # Returns a string with the languages, sorted by their frequency

    def __init__(self, seed=0):
        self.seed = seed

    @staticmethod
    def metadata_name() -> str:
        return 'language'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    @staticmethod
    def detect_languages(text):
        if len(text) == 0:
            detect(text)  # trigger LangDetectException. Throwing one in here somehow doesnt work
        if DelimiterTypeMetadata().metadata_function(text) == 'HTML':
            parts = re.split(r'<\s*br\s*/?>', text)
        else:
            parts = re.split(r'[\n\r]+', text)
        parts = [x.strip() for x in parts if x.strip()]
        detected_languages = defaultdict(int)
        for part in parts:
            lang = detect(part)
            detected_languages[lang] += 1
        return detected_languages

    def metadata_function(self, text, seed=0):
        DetectorFactory.seed = self.seed
        return dictionary_to_sorted_string(self.detect_languages(text))


class LanguageMetadata(GenericTextMetadata):

    def __init__(self, seed=0):
        self.seed = seed

    @staticmethod
    def metadata_name() -> str:
        return 'language'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.categorical

    def metadata_function(self, text):
        DetectorFactory.seed = self.seed
        return detect(text)


class ComplexityMetadata(GenericTextMetadata):

    @staticmethod
    def metadata_name() -> str:
        return 'complexity'

    def metadata_column_type(self) -> ColumnType:
        return ColumnType.numerical

    def metadata_function(self, text):
        # works best for longer english texts. kinda works for other languages as well (not good though)
        return textstat.text_standard(text, True)


class TextMetadata(Precalculation):

    def __init__(self, text_metadata_types=None, language='en', infer_language=False):
        if text_metadata_types is None:
            self.text_metadata_types = frozenset([NumCharsMetadata(), NumWordsMetadata(), DistinctWordsRatioMetadata()])
        else:
            self.text_metadata_types = frozenset(text_metadata_types)
        if infer_language or language != 'en':
            for mdtype in self.text_metadata_types:
                try:
                    mdtype.language = language
                    mdtype.infer_language = infer_language
                except AttributeError:
                    continue  # do nothing for types which do not accept a language as parameter

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