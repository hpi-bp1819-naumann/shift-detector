import unittest

from pandas.util.testing import assert_frame_equal

import morpheus.utils.text_metadata_utils as TmUtils
from langdetect.lang_detect_exception import LangDetectException

from morpheus.precalculations.store import Store
from morpheus.precalculations.text_metadata import *

import tests.test_data as td


class TestTextMetadataPrecalculations(unittest.TestCase):

    def setUp(self):
        df1 = pd.DataFrame.from_dict({'text': td.poems})
        df2 = pd.DataFrame.from_dict({'text': td.phrases})
        self.store = Store(df1, df2)

    def test_num_chars(self):
        md1, md2 = NumCharsMetadata().process(self.store)
        solution1 = pd.DataFrame([132, 123, 117, 136, 137, 133, 149, 92, 79, 86], columns=['text'])
        solution2 = pd.DataFrame([31, 35, 27, 28, 34, 40, 34, 32, 23, 26], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_ratio_upper(self):
        md1, md2 = RatioUppercaseLettersMetadata().process(self.store)
        solution1 = pd.DataFrame([0.0392156, 0.0537634, 0.0439560, 0.0380952, 0.0370370, 0.08, 0.0434782, 0.0563380,
                                  0.0317460, 0.03125], columns=['text'])
        solution2 = pd.DataFrame([0.0370370, 0.03125, 0.04, 0.04, 0.0322580, 0.0277777, 0.0333333, 0.0344827,
                                  0.0588235, 0.0416666], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unicode_categories(self):
        md1, md2 = UnicodeCategoriesMetadata().process(self.store)
        solution1 = pd.DataFrame(['Ll, Zs, Po, Lu, Cc', 'Ll, Zs, Po, Lu, Cc', 'Ll, Zs, Po, Lu, Cc',
                                  'Ll, Zs, Po, Lu, Cc', 'Ll, Zs, Po, Lu, Cc', 'Ll, Zs, Lu, Po, Cc, Pf, Pd',
                                  'Ll, Zs, Lu, Po, Cc, Pd', 'Ll, Zs, Po, Lu, Cc', 'Ll, Zs, Lu, Cc, Po',
                                  'Ll, Zs, Po, Lu, Cc'], columns=['text'])
        solution2 = pd.DataFrame(['Ll, Zs, Lu, Pd', 'Ll, Zs, Lu, Pd', 'Ll, Zs, Lu', 'Ll, Zs, Lu', 'Ll, Zs, Lu',
                                  'Ll, Zs, Lu, Pd', 'Ll, Zs, Lu, Nd', 'Ll, Zs, Lu, Pd', 'Ll, Nd, Zs, Lu, Po',
                                  'Ll, Zs, Lu'], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unicode_blocks(self):
        md1, md2 = UnicodeBlocksMetadata().process(self.store)
        solution1 = pd.DataFrame([23, 22], columns=['text'])
        solution2 = pd.DataFrame([5, 4], columns=['text'])
        solution1 = pd.DataFrame(['Basic Latin', 'Basic Latin', 'Basic Latin', 'Basic Latin', 'Basic Latin',
                                  'Basic Latin, General Punctuation', 'Basic Latin', 'Basic Latin', 'Basic Latin',
                                  'Basic Latin'], columns=['text'])
        solution2 = pd.DataFrame(['Basic Latin'] * 10, columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_num_words(self):
        md1, md2 = NumWordsMetadata().process(self.store)
        solution1 = pd.DataFrame([26, 25, 22, 27, 24, 23, 28, 14, 16, 18], columns=['text'])
        solution2 = pd.DataFrame([5, 4, 3, 4, 4, 5, 4, 4, 3, 3], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_distinct_words(self):
        md1, md2 = DistinctWordsRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([24/26, 19/25, 21/22, 24/27, 20/24, 20/23, 25/28, 14/14, 15/16, 16/18],
                                 columns=['text'])
        solution2 = pd.DataFrame([1.0] * 10, columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unique_words(self):
        md1, md2 = UniqueWordsRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([22/26, 14/25, 20/22, 21/27, 19/24, 17/23, 23/28, 14/14, 14/16, 14/18],
                                 columns=['text'])
        solution2 = pd.DataFrame([1.0] * 10, columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unknown_ratio(self):
        md1, md2 = UnknownWordRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([0.0, 0.04, 0.0, 0.0, 0.0, 0.0869565, 0.0, 0.0714285, 0.0, 0.0555555],
                                 columns=['text'])
        solution2 = pd.DataFrame([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_stopword_ratio(self):
        md1, md2 = StopwordRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([0.576923, 0.48, 0.5, 0.4444444, 0.4166666, 0.2608695, 0.4642857, 0.2857142, 0.625,
                                  0.5], columns=['text'])
        solution2 = pd.DataFrame([0.0] * 10, columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_delimiter_type(self):
        md1, md2 = DelimiterTypeMetadata().process(self.store)
        solution1 = pd.DataFrame(['newline'] * 10, columns=['text'])
        solution2 = pd.DataFrame(['whitespace'] * 10, columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_num_parts(self):
        md1, md2 = NumPartsMetadata().process(self.store)
        solution1 = pd.DataFrame([4, 4, 4, 4, 4, 4, 4, 4, 2, 2], columns=['text'])
        solution2 = pd.DataFrame([4, 3, 3, 4, 4, 4, 4, 3, 3, 3], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_language(self):
        md1, md2 = LanguagePerParagraph().process(self.store)
        solution1 = pd.DataFrame(['en'] * 10, columns=['text'])
        solution2 = pd.DataFrame(['en'] * 10, columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    # seems to be dependent on the machine: travis gets different results
    # def test_complexity(self):
    #    md1, md2 = ComplexityMetadata().process(self.store)
    #    solution1 = pd.DataFrame([5.0, 3.0], columns=['text'])
    #    solution2 = pd.DataFrame([0.0, 13.0], columns=['text'])
    #    assert_frame_equal(solution1, md1)
    #    assert_frame_equal(solution2, md2)

    def test_pos_tags(self):
        md1, md2 = PartOfSpeechMetadata().process(self.store)
        solution1 = pd.DataFrame(['NOUN, ., VERB, ADJ, ADP, PRON, ADV, CONJ, DET',
                                  'NOUN, ., VERB, ADV, ADJ, DET, ADP, CONJ, PRON, PRT'], columns=['text'])
        solution2 = pd.DataFrame(['NOUN, ADJ, VERB', 'ADJ, NOUN'], columns=['text'])
        assert_frame_equal(solution1, md1.iloc[:2, :])
        assert_frame_equal(solution2, md2.iloc[:2, :])

    def test_metadata_precalculation(self):
        md1, md2 = self.store[TextMetadata(text_metadata_types=[NumWordsMetadata(), StopwordRatioMetadata(),
                                                                UnicodeBlocksMetadata()])]
        index = pd.MultiIndex.from_product([['text'], ['num_words', 'stopword_ratio', 'unicode_blocks']],
                                           names=['column', 'metadata'])
        solution1 = pd.DataFrame(columns=index)
        solution2 = pd.DataFrame(columns=index)
        solution1[('text', 'num_words')] = [26, 25, 22, 27, 24, 23, 28, 14, 16, 18]
        solution2[('text', 'num_words')] = [5, 4, 3, 4, 4, 5, 4, 4, 3, 3]
        solution1[('text', 'stopword_ratio')] = [0.576923, 0.48, 0.5, 0.4444444, 0.4166666, 0.2608695, 0.4642857,
                                                 0.2857142, 0.625, 0.5]
        solution2[('text', 'stopword_ratio')] = [0.0] * 10
        solution1[('text', 'unicode_blocks')] = ['Basic Latin', 'Basic Latin', 'Basic Latin', 'Basic Latin',
                                                 'Basic Latin', 'Basic Latin, General Punctuation', 'Basic Latin',
                                                 'Basic Latin', 'Basic Latin', 'Basic Latin']
        solution2[('text', 'unicode_blocks')] = ['Basic Latin'] * 10
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)


class TestTextMetadataFunctions(unittest.TestCase):

    def setUp(self):
        self.empty_string = ""
        self.punctuation_string = "., \t \n !`"
        self.english_string = "This is a normal sentence. This is for testing."
        self.lower_string = "justlowerletters"
        self.upper_string = "ALL UPPER CASE LETTERS"
        self.unicode_string = "\n \u0600 \uF8FF \uDB80 Hi \u02B7 \u0C99 \u1F8D \u094A \uA670 ∑ ´ 42 \u2169 " \
                              "‚·°‡ﬁ›‹€⁄¡™£¢∞§¶•ªº‘«»æ…ÆÚ˘¯≤≤≥ ,;' "
        self.german_string = "Dies ist ein einfacher Satz."
        self.html_string = "the text is well written. <p> some other very good text < br/ > "\
                           "Aber es gibt auch deutschen Text"
        self.comma_string = "some text, some other text -- more text."
        self.whitespace_string = "some text some other text more text"
        self.html_sentence_string = "some text <p> some other text. more text."
        self.sentence_other_string = "some text, some other text. more text."
        self.html_sentence_other_string = "some text -- some. other text <br> more text."
        self.multiple_languages_string = "Dieser Text ist zum Teil deutsch. \n Part of this text is in english. \n "\
                                         "there actually is some french coming. \n Ce n'est pas anglais. \n "\
                                         "No puedo hablar español. \n Beberapa bahasa untuk diuji."
        self.incorrect_english_string = "Thhis is a nirnal sentense. Lanquage detecction is esay."

        self.empty_array = []
        self.distinct_words_array = ['this', 'are', 'all', 'different', 'words']
        self.same_words_array = ['same', 'same', 'same', 'same']
        self.mixed_words_array = ['there', 'are', 'doubled', 'words', 'and', 'there', 'are', 'distinct', 'words']
        self.english_array = ['this', 'is', 'a', 'correct', 'sentence']
        self.incorrect_english_array = ['thiis', 'is', 'an', 'incozyzyrrect', 'sentence']
        self.no_stopwords_array = ['computer', 'calculates', 'math']
        self.only_stopwords_array = ['the', 'and', 'is', 'i', 'am']
        self.french_array = ['demain', 'dès', 'l’aube', 'à', 'l’heure', 'où', 'blanchit', 'la', 'campagne', 'je',
                             'partirai', 'vois', 'tu', 'je', 'sais', 'que', 'tu', 'm’attends', 'j’irai', 'par', 'la',
                             'forêt', 'j’irai', 'par', 'la', 'montagne', 'je', 'ne', 'puis', 'demeurer', 'loin', 'de',
                             'toi', 'plus', 'longtemps']
        self.unsupported_language_array = ['aqoonyahanada', 'caalamku', 'waxay', 'aad', 'ugu', 'murmaan', 'sidii',
                                           'luuqadaha', 'aduunku', 'ku', 'bilaabmeem']

        self.empty_dict = {}
        self.many_entries_dict = {'a': 2, 'b': 5, 'c': 3, 'f': 5, 'd': 1, 'e': 5}
        self.one_entry_dict = {'a': 100}

    def test_dictionary_to_sorted_string(self):
        self.assertEqual(TmUtils.dictionary_to_sorted_string(self.many_entries_dict), "b, e, f, c, a, d")
        self.assertEqual(TmUtils.dictionary_to_sorted_string(self.one_entry_dict), "a")
        self.assertEqual(TmUtils.dictionary_to_sorted_string(self.empty_dict), "")

    def test_num_chars(self):
        num_chars = NumCharsMetadata().metadata_function
        self.assertEqual(num_chars(self.english_string), 47)
        self.assertEqual(num_chars(self.unicode_string), 66)
        self.assertEqual(num_chars(self.punctuation_string), 9)
        self.assertEqual(num_chars(self.empty_string), 0)

    def test_ratio_upper(self):
        ratio_upper = RatioUppercaseLettersMetadata().metadata_function
        self.assertEqual(ratio_upper(self.lower_string), 0.00)
        self.assertEqual(ratio_upper(self.upper_string), 1.00)
        self.assertAlmostEqual(ratio_upper(self.english_string), 0.05405405)
        self.assertEqual(ratio_upper(self.empty_string), 0.00)

    def test_unicode_category(self):
        unicode_category_histogram = UnicodeCategoriesMetadata().unicode_category_histogram
        self.assertEqual(unicode_category_histogram(self.lower_string), {'Ll': 16})
        self.assertEqual(unicode_category_histogram(self.unicode_string), {'Cc': 1, 'Zs': 16, 'Cf': 1, 'Co': 1, 'Cs': 1,
                                                                           'Lu': 3, 'Ll': 3, 'Lm': 1, 'Lo': 3, 'Lt': 1,
                                                                           'Nl': 1, 'Ps': 1, 'Po': 10, 'Mc': 1, 'Me': 1,
                                                                           'Sm': 6, 'Sk': 3, 'Nd': 2, 'So': 2, 'Pf': 2,
                                                                           'Pi': 3, 'Sc': 3})
        self.assertEqual(unicode_category_histogram(self.empty_string), {})

    def test_unicode_block(self):
        latin = "Latin Letters! *with punctuation,!./ and numbers 983"
        unicode_block_histogram = UnicodeBlocksMetadata().unicode_block_histogram
        self.assertEqual(unicode_block_histogram(latin), {'Basic Latin': 52})
        self.assertEqual(unicode_block_histogram(self.unicode_string), {'Basic Latin': 24, 'Arabic': 1,
                                                                        'Private Use Area': 1,
                                                                        'High Private Use Surrogates': 1,
                                                                        'Spacing Modifier Letters': 2, 'Kannada': 1,
                                                                        'Greek Extended': 1, 'Devanagari': 1,
                                                                        'Cyrillic Extended-B': 1,
                                                                        'Mathematical Operators': 5,
                                                                        'Latin-1 Supplement': 16, 'Number Forms': 1,
                                                                        'General Punctuation': 8,
                                                                        'Alphabetic Presentation Forms': 1,
                                                                        'Currency Symbols': 1, 'Letterlike Symbols': 1})
        self.assertEqual(unicode_block_histogram(self.empty_string), {})

    def test_num_words(self):
        num_words = NumWordsMetadata().metadata_function
        self.assertEqual(num_words(self.distinct_words_array), 5)
        self.assertEqual(num_words(self.same_words_array), 4)
        self.assertEqual(num_words(self.empty_array), 0)

    def test_distinct_words_ratio(self):
        distinct_words_ratio = DistinctWordsRatioMetadata().metadata_function
        self.assertEqual(distinct_words_ratio(self.distinct_words_array), 1.0)
        self.assertEqual(distinct_words_ratio(self.same_words_array), 0.25)
        self.assertAlmostEqual(distinct_words_ratio(self.mixed_words_array), 0.66666666)
        self.assertEqual(distinct_words_ratio(self.empty_array), 0.0)

    def test_unique_words(self):
        unique_words_ratio = UniqueWordsRatioMetadata().metadata_function
        self.assertEqual(unique_words_ratio(self.distinct_words_array), 1.0)
        self.assertEqual(unique_words_ratio(self.same_words_array), 0.0)
        self.assertAlmostEqual(unique_words_ratio(self.mixed_words_array), 0.3333333)
        self.assertEqual(unique_words_ratio(self.empty_array), 0.0)

    def test_unknown_words(self):
        unknown_word_ratio = UnknownWordRatioMetadata().metadata_function
        self.assertEqual(unknown_word_ratio('en', self.english_array), 0.00)
        self.assertEqual(unknown_word_ratio('en', self.incorrect_english_array), 0.4)
        self.assertRaises(ValueError, unknown_word_ratio, language='so', words=self.unsupported_language_array)
        self.assertAlmostEqual(unknown_word_ratio('fr', self.french_array), 0.1142857, places=5)
        self.assertEqual(unknown_word_ratio('en', self.empty_array), 00.00)

    def test_stopwords(self):
        stopword_ratio = StopwordRatioMetadata().metadata_function
        self.assertEqual(stopword_ratio('en', self.no_stopwords_array), 0.0)
        self.assertEqual(stopword_ratio('en', self.only_stopwords_array), 1.0)
        self.assertEqual(stopword_ratio('en', self.english_array), 0.6)
        self.assertAlmostEqual(stopword_ratio('fr', self.french_array), 0.4285714, places=5)
        self.assertEqual(stopword_ratio('en', self.empty_array), 0.0)
        self.assertRaises(ValueError, stopword_ratio, language='so', words=self.unsupported_language_array)

    def test_category(self):
        delimiter_type = DelimiterTypeMetadata().metadata_function
        self.assertEqual(delimiter_type(self.html_string), "HTML")
        self.assertEqual(delimiter_type(self.english_string), "sentence")
        self.assertEqual(delimiter_type(self.comma_string), "comma")
        self.assertEqual(delimiter_type(self.whitespace_string), "whitespace")
        self.assertEqual(delimiter_type(self.html_sentence_string), "HTML")
        self.assertEqual(delimiter_type(self.sentence_other_string), "sentence")
        self.assertEqual(delimiter_type(self.html_sentence_other_string), "HTML")
        self.assertEqual(delimiter_type(self.empty_string), "no delimiter")

    def test_num_parts(self):
        num_parts = NumPartsMetadata().metadata_function
        self.assertEqual(num_parts(self.html_string), 3)
        self.assertEqual(num_parts(self.english_string), 2)
        self.assertEqual(num_parts(self.comma_string), 2)
        self.assertEqual(num_parts(self.whitespace_string), 7)
        self.assertEqual(num_parts(self.html_sentence_string), 2)
        self.assertEqual(num_parts(self.sentence_other_string), 2)
        self.assertEqual(num_parts(self.html_sentence_other_string), 2)
        self.assertEqual(num_parts(self.empty_string), 0)

    def test_languages(self):
        language = LanguagePerParagraph().detect_languages
        self.assertEqual(language(self.english_string), {'en': 1})
        self.assertEqual(language(self.incorrect_english_string), {'en': 1})
        self.assertEqual(language(self.german_string), {'de': 1})
        self.assertEqual(language(self.html_string), {'en': 1, 'de': 1})
        self.assertEqual(language(self.multiple_languages_string), {'en': 2, 'de': 1, 'fr': 1, 'es': 1, 'id': 1})
        self.assertRaises(LangDetectException, language, text=self.punctuation_string)
        self.assertRaises(LangDetectException, language, text=self.empty_string)

    def test_complexity(self):
        # hard = "Quantum mechanics (QM; also known as quantum physics, quantum theory, the wave mechanical model, "\
        #       "or matrix mechanics), including quantum field theory, is a fundamental theory in physics which "\
        #       "describes nature at the smallest scales of energy levels of atoms and subatomic particles."
        text_complexity = ComplexityMetadata().metadata_function
        self.assertEqual(text_complexity('en', self.empty_string), 0.0)
        self.assertEqual(text_complexity('en', self.punctuation_string), 0.0)
        self.assertEqual(text_complexity('en', self.english_string), text_complexity('en', self.english_string))
        self.assertRaises(ValueError, text_complexity, language='de', text=self.german_string)
        # Works in Travis for Python 3.6 but not for 3.5. 3.5 seems to not support the complexity metric.
        # self.assertGreater(text_complexity(hard), text_complexity(easy))

    def test_pos_tags(self):
        pos_tags = PartOfSpeechMetadata().metadata_function
        self.assertEqual('DET, VERB, ., ADJ, ADP, NOUN', pos_tags('en', self.english_string))
        self.assertEqual('.', pos_tags('en', self.punctuation_string))
        self.assertEqual('', pos_tags('en', self.empty_string))
        self.assertRaises(ValueError, pos_tags, language='de', text=self.german_string)
