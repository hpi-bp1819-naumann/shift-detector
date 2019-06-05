import unittest

from pandas.util.testing import assert_frame_equal

import shift_detector.utils.TextMetadataUtils as TmUtils
from langdetect.lang_detect_exception import LangDetectException

from shift_detector.precalculations.Store import Store
from shift_detector.precalculations.TextMetadata import *


class TestTextMetadataPreprocessors(unittest.TestCase):

    def setUp(self):
        poems = [
            'Tell me not, in mournful numbers,\nLife is but an empty dream!\nFor the soul is dead that slumbers,\nAnd things are not what they seem.',
            'Life is real! Life is earnest!\nAnd the grave is not its goal;\nDust thou art, to dust returnest,\nWas not spoken of the soul.'
        ]
        phrases = [
            'Front-line leading edge website',
            'Upgradable upward-trending software'
        ]
        df1 = pd.DataFrame.from_dict({'text': poems})
        df2 = pd.DataFrame.from_dict({'text': phrases})
        self.store = Store(df1, df2)

    def test_tokenize_into_words(self):
        md1, md2 = TokenizeIntoWords().process(self.store)
        sol1_1 = ['Tell', 'me', 'not', 'in', 'mournful', 'numbers', 'Life', 'is', 'but', 'an', 'empty', 'dream', 'For', 'the', 'soul', 'is', 'dead', 'that', 'slumbers', 'And', 'things', 'are', 'not', 'what', 'they', 'seem']
        sol1_2 = ['Life', 'is', 'real', 'Life', 'is', 'earnest', 'And', 'the', 'grave', 'is', 'not', 'its', 'goal', 'Dust', 'thou', 'art', 'to', 'dust', 'returnest', 'Was', 'not', 'spoken', 'of', 'the', 'soul']
        sol2_1 = ['Front', 'line', 'leading', 'edge', 'website']
        sol2_2 = ['Upgradable', 'upward', 'trending', 'software']
        solution1 = pd.DataFrame([[sol1_1], [sol1_2]], columns=['text'])
        solution2 = pd.DataFrame([[sol2_1], [sol2_2]], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_num_chars(self):
        md1, md2 = NumCharsMetadata().process(self.store)
        solution1 = pd.DataFrame([132, 123], columns=['text'])
        solution2 = pd.DataFrame([31, 35], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_ratio_upper(self):
        md1, md2 = RatioUppercaseLettersMetadata().process(self.store)
        solution1 = pd.DataFrame([0.0392156, 0.0537634], columns=['text'])
        solution2 = pd.DataFrame([0.0370370, 0.03125], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unicode_categories(self):
        md1, md2 = UnicodeCategoriesMetadata().process(self.store)
        solution1 = pd.DataFrame(['Ll, Zs, Po, Lu, Cc', 'Ll, Zs, Po, Lu, Cc'], columns=['text'])
        solution2 = pd.DataFrame(['Ll, Zs, Lu, Pd', 'Ll, Zs, Lu, Pd'], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unicode_blocks(self):
        md1, md2 = UnicodeBlocksMetadata().process(self.store)
        solution1 = pd.DataFrame([23, 22], columns=['text'])
        solution2 = pd.DataFrame([5, 4], columns=['text'])
        solution1 = pd.DataFrame(['Basic Latin', 'Basic Latin'], columns=['text'])
        solution2 = pd.DataFrame(['Basic Latin', 'Basic Latin'], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_num_words(self):
        md1, md2 = NumWordsMetadata().process(self.store)
        solution1 = pd.DataFrame([26, 25], columns=['text'])
        solution2 = pd.DataFrame([5, 4], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_distinct_words(self):
        md1, md2 = DistinctWordsRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([24/26, 20/25], columns=['text'])
        solution2 = pd.DataFrame([1.0, 1.0], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unique_words(self):
        md1, md2 = UniqueWordsRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([22/26, 16/25], columns=['text'])
        solution2 = pd.DataFrame([1.0, 1.0], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_unknown_ratio(self):
        md1, md2 = UnknownWordRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([0.0, 0.04], columns=['text'])
        solution2 = pd.DataFrame([0.2, 0.0], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_stopword_ratio(self):
        md1, md2 = StopwordRatioMetadata().process(self.store)
        solution1 = pd.DataFrame([0.576923, 0.48], columns=['text'])
        solution2 = pd.DataFrame([0.0, 0.0], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_delimiter_type(self):
        md1, md2 = DelimiterTypeMetadata().process(self.store)
        solution1 = pd.DataFrame(['newline', 'newline'], columns=['text'])
        solution2 = pd.DataFrame(['whitespace', 'whitespace'], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_num_parts(self):
        md1, md2 = NumPartsMetadata().process(self.store)
        solution1 = pd.DataFrame([4, 4], columns=['text'])
        solution2 = pd.DataFrame([4, 3], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_language(self):
        md1, md2 = LanguagePerParagraph().process(self.store)
        solution1 = pd.DataFrame(['en', 'en'], columns=['text'])
        solution2 = pd.DataFrame(['en', 'en'], columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    # seems to be dependent on the machine: travis gets different results
    #def test_complexity(self):
    #    md1, md2 = ComplexityMetadata().process(self.store)
    #    solution1 = pd.DataFrame([5.0, 3.0], columns=['text'])
    #    solution2 = pd.DataFrame([0.0, 13.0], columns=['text'])
    #    assert_frame_equal(solution1, md1)
    #    assert_frame_equal(solution2, md2)

    def test_metadata_preprocessor(self):
        md1, md2 = self.store[TextMetadata(text_metadata_types=[NumWordsMetadata(), StopwordRatioMetadata(), UnicodeBlocksMetadata()])]
        index = pd.MultiIndex.from_product([['text'], ['num_words', 'stopword_ratio', 'unicode_blocks']], names=['column', 'metadata'])
        solution1 = pd.DataFrame(columns=index)
        solution2 = pd.DataFrame(columns=index)
        solution1[('text', 'num_words')] = [26, 25]
        solution2[('text', 'num_words')] = [5, 4]
        solution1[('text', 'stopword_ratio')] = [0.576923, 0.480000]
        solution2[('text', 'stopword_ratio')] = [0.000000, 0.000000]
        solution1[('text', 'unicode_blocks')] = ['Basic Latin', 'Basic Latin']
        solution2[('text', 'unicode_blocks')] = ['Basic Latin', 'Basic Latin']
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)


class TestTextMetadataFunctions(unittest.TestCase):

    def test_tokenize_into_words(self):
        normal = "This. is a'n example, ,, 12  35,6  , st/r--ing    \n test."
        empty = ""
        punctuation = ".  , * (  \n \t [}"
        tokenize_into_words = TokenizeIntoWords().tokenize_into_words
        self.assertEqual(tokenize_into_words(normal), ['This', 'is', "a'n", 'example', '12', '356', 'str', 'ing', 'test'])
        self.assertEqual(tokenize_into_words(empty), [])
        self.assertEqual(tokenize_into_words(punctuation), [])

    def test_dictionary_to_sorted_string(self):
        many = {'a': 2, 'b': 5, 'c': 3, 'f': 5, 'd': 1, 'e': 5} 
        one = {'a': 100}
        empty = {}
        self.assertEqual(TmUtils.dictionary_to_sorted_string(many), "b, e, f, c, a, d")
        self.assertEqual(TmUtils.dictionary_to_sorted_string(one), "a")
        self.assertEqual(TmUtils.dictionary_to_sorted_string(empty), "")

    def test_num_chars(self):
        normal = "normaler Text"
        unicodes = "\u6667 is one char"
        punctuation = "., < \t \n !`"
        empty = ""
        num_chars = NumCharsMetadata().metadata_function
        self.assertEqual(num_chars(normal), 13)
        self.assertEqual(num_chars(unicodes), 13)
        self.assertEqual(num_chars(punctuation), 11)
        self.assertEqual(num_chars(empty), 0)

    def test_ratio_upper(self):
        lower = "no upper case letters"
        upper = "ALL UPPER CASE LETTERS"
        mixed1 = "FifTY fIFty"
        mixed2 = "Tre"
        empty = ""
        ratio_upper = RatioUppercaseLettersMetadata().metadata_function
        self.assertEqual(ratio_upper(lower), 0.00)
        self.assertEqual(ratio_upper(upper), 1.00)
        self.assertEqual(ratio_upper(mixed1), .50)
        self.assertAlmostEqual(ratio_upper(mixed2), 0.3333333)
        self.assertEqual(ratio_upper(empty), 0.00)

    def test_unicode_category(self):
        lower = "justlowerletters"
        different = "\n \u0600 \uF8FF \uDB80 Hi \u02B7 \u0C99 \u1F8D \u094A \uA670 ∑ ´ 42 \u2169 ‚·°‡ﬁ›‹€⁄¡™£¢∞§¶•ªº‘«»æ…ÆÚ˘¯≤≤≥ ,;' "
        empty = ""
        unicode_category_histogram = UnicodeCategoriesMetadata().unicode_category_histogram
        self.assertEqual(unicode_category_histogram(lower), {'Ll': 16})
        self.assertEqual(unicode_category_histogram(different), {'Cc': 1, 'Zs': 16, 'Cf': 1, 'Co': 1, 'Cs': 1, 'Lu': 3, 'Ll': 3, 'Lm': 1, 'Lo': 3, 'Lt': 1, 'Mc': 1, 'Me': 1, 'Sm': 6, 'Sk': 3, 'Nd': 2, 'Nl': 1, 'Ps': 1, 'Po': 10, 'So': 2, 'Pf': 2, 'Pi': 3, 'Sc': 3})
        self.assertEqual(unicode_category_histogram(empty), {})

    def test_unicode_block(self):
        latin = "Latin Letters! *with punctuation,!./ and numbers 983"
        different = "\n \u0600 \uF8FF \uDB80 Hi \u02B7 \u0C99 \u1F8D \u094A \uA670 ∑ ´ 42 \u2169 ‚·°‡ﬁ›‹€⁄¡™£¢∞§¶•ªº‘«»æ…ÆÚ˘¯≤≤≥ ,;' "
        empty = ""
        unicode_block_histogram = UnicodeBlocksMetadata().unicode_block_histogram
        self.assertEqual(unicode_block_histogram(latin), {'Basic Latin': 52})
        self.assertEqual(unicode_block_histogram(different), {'Basic Latin': 24, 'Arabic': 1, 'Private Use Area': 1, 'High Private Use Surrogates': 1, 'Spacing Modifier Letters': 2, 'Kannada': 1, 'Greek Extended': 1, 'Devanagari': 1, 'Cyrillic Extended-B': 1, 'Mathematical Operators': 5, 'Latin-1 Supplement': 16, 'Number Forms': 1, 'General Punctuation': 8, 'Alphabetic Presentation Forms': 1, 'Currency Symbols': 1, 'Letterlike Symbols': 1})
        self.assertEqual(unicode_block_histogram(empty), {})

    def test_num_words(self):
        distinct = ['this', 'are', 'all', 'different', 'words']
        same = ['same', 'same', 'same', 'same']
        empty = []
        num_words = NumWordsMetadata().metadata_function
        self.assertEqual(num_words(distinct), 5)
        self.assertEqual(num_words(same), 4)
        self.assertEqual(num_words(empty), 0)

    def test_distinct_words_ratio(self):
        distinct = ['this', 'are', 'all', 'different', 'words']
        same = ['same', 'same', 'same', 'same']
        mixed = ['there', 'are', 'doubled', 'words', 'and', 'there', 'are', 'distinct', 'words']
        capitalLetters = ['Capital', 'letters', 'matter', 'Matter']
        empty = []
        distinct_words_ratio = DistinctWordsRatioMetadata().metadata_function
        self.assertEqual(distinct_words_ratio(distinct), 1.0)
        self.assertEqual(distinct_words_ratio(same), 0.25)
        self.assertAlmostEqual(distinct_words_ratio(mixed), 0.66666666)
        self.assertEqual(distinct_words_ratio(capitalLetters), 1.0)
        self.assertEqual(distinct_words_ratio(empty), 0.0)

    def test_unique_words(self):
        distinct = ['this', 'are', 'all', 'different', 'words']
        same = ['same', 'same', 'same', 'same']
        mixed = ['there', 'are', 'doubled', 'words', 'and', 'there', 'are', 'distinct', 'words']
        capitalLetters = ['Capital', 'letters', 'letters', 'matter', 'Matter']
        empty = []
        unique_words_ratio = UniqueWordsRatioMetadata().metadata_function
        self.assertEqual(unique_words_ratio(distinct), 1.0)
        self.assertEqual(unique_words_ratio(same), 0.0)
        self.assertAlmostEqual(unique_words_ratio(mixed), 0.3333333)
        self.assertEqual(unique_words_ratio(capitalLetters), 0.6)
        self.assertEqual(unique_words_ratio(empty), 0.0)

    def test_unknown_words(self):
        correct_english = ('en', ['This', 'is', 'a', 'correct', 'sentence'])
        incorrect_english = ('en', ['Thiis', 'is', 'an', 'incozyzyrrect', 'sentence'])
        french = ('fr', ['Demain', 'dès', 'l’aube', 'à', 'l’heure', 'où', 'blanchit', 'la', 'campagne', 'Je', 'partirai', 'Vois', 'tu', 'je', 'sais', 'que', 'tu', 'm’attends', 'J’irai', 'par', 'la', 'forêt', 'j’irai', 'par', 'la', 'montagne', 'Je', 'ne', 'puis', 'demeurer', 'loin', 'de', 'toi', 'plus', 'longtemps'])
        empty = ('en', [])
        unsupported_language = ('so', ['Aqoonyahanada', 'caalamku', 'waxay', 'aad', 'ugu', 'murmaan', 'sidii', 'luuqadaha', 'aduunku', 'ku', 'bilaabmeem'])
        unknown_word_ratio = UnknownWordRatioMetadata().metadata_function
        self.assertEqual(unknown_word_ratio(correct_english), 0.00)
        self.assertEqual(unknown_word_ratio(incorrect_english), .4)
        self.assertRaises(ValueError, unknown_word_ratio, language_and_words=unsupported_language)
        self.assertAlmostEqual(unknown_word_ratio(french), 0.1142857, places=5)
        self.assertEqual(unknown_word_ratio(empty), 00.00)

    def test_stopwords(self):
        no_stopwords = ('en', ['computer', 'calculates', 'math'])
        only_stopwords = ('en', ['The', 'and', 'is', 'I', 'am'])
        mixed = ('en', ['A', 'normal', 'sentence', 'has', 'both'])
        french = ('fr', ['Demain', 'dès', 'l’aube', 'à', 'l’heure', 'où', 'blanchit', 'la', 'campagne', 'Je', 'partirai', 'Vois', 'tu', 'je', 'sais', 'que', 'tu', 'm’attends', 'J’irai', 'par', 'la', 'forêt', 'j’irai', 'par', 'la', 'montagne', 'Je', 'ne', 'puis', 'demeurer', 'loin', 'de', 'toi', 'plus', 'longtemps'])
        empty = ('en', [])
        unsupported_language = ('so', ['Aqoonyahanada', 'caalamku', 'waxay', 'aad', 'ugu', 'murmaan', 'sidii', 'luuqadaha', 'aduunku', 'ku', 'bilaabmeem'])
        stopword_ratio = StopwordRatioMetadata().metadata_function
        self.assertEqual(stopword_ratio(no_stopwords), 0.0)
        self.assertEqual(stopword_ratio(only_stopwords), 1.0)
        self.assertEqual(stopword_ratio(mixed), 0.6)
        self.assertAlmostEqual(stopword_ratio(french), 0.4285714, places=5)
        self.assertEqual(stopword_ratio(empty), 0.0)
        self.assertRaises(ValueError, stopword_ratio, language_and_words=unsupported_language)


    def test_category(self):
        html = "some text <p> some other text < br/ > more text"
        sentence = "some text. some other text. more text."
        comma = "some text, some other text -- more text."
        white = "some text some other text more text"
        htmlsentence = "some text <p> some other text. more text."
        sentenceother = "some text, some other text. more text."
        htmlsentenceother = "some text -- some. other text <br> more text."
        empty = ""
        delimiter_type = DelimiterTypeMetadata().metadata_function
        self.assertEqual(delimiter_type(html), "HTML")
        self.assertEqual(delimiter_type(sentence), "sentence")
        self.assertEqual(delimiter_type(comma), "comma")
        self.assertEqual(delimiter_type(white), "whitespace")
        self.assertEqual(delimiter_type(htmlsentence), "HTML")
        self.assertEqual(delimiter_type(sentenceother), "sentence")
        self.assertEqual(delimiter_type(htmlsentenceother), "HTML")
        self.assertEqual(delimiter_type(empty), "no delimiter")

    def test_num_parts(self):
        html = "some text <p> some other text < br/ > more text"
        sentence = "some text. some other text. more text."
        comma = "some text, some other text -- more text."
        none = "some text some other text more text"
        htmlsentence = "some text <p> some other text. more text."
        sentenceother = "some text, some other text. more text."
        htmlsentenceother = "some text -- some. other text <br> more text."
        empty = ""
        num_parts = NumPartsMetadata().metadata_function
        self.assertEqual(num_parts(html), 3)
        self.assertEqual(num_parts(sentence), 3)
        self.assertEqual(num_parts(comma), 2)
        self.assertEqual(num_parts(none), 7)
        self.assertEqual(num_parts(htmlsentence), 2)
        self.assertEqual(num_parts(sentenceother), 2)
        self.assertEqual(num_parts(htmlsentenceother), 2)
        self.assertEqual(num_parts(empty), 0)

    def test_languages(self):
        english = "This is a normal sentence. Language detection is easy."
        englishTypos = "Thhis is a nirnal sentense. Lanquage detecction is esay."
        german = "Dies ist ein einfacher Satz. Kurz und knackig."
        englishgerman = "Dieser Text ist zum Teil deutsch. <br> Part of this text is in english"
        multipleLanguages = "Dieser Text ist zum Teil deutsch. \n Part of this text is in english. \n there actually is some french coming. \n Ce n'est pas anglais. \n No puedo hablar español. \n Beberapa bahasa untuk diuji."
        punctuation = " . ,"
        empty = ""
        language = LanguagePerParagraph().detect_languages
        self.assertEqual(language(english), {'en': 1})
        self.assertEqual(language(englishTypos), {'en': 1})
        self.assertEqual(language(german), {'de': 1})
        self.assertEqual(language(englishgerman), {'en': 1, 'de': 1})
        self.assertEqual(language(multipleLanguages), {'en': 2, 'de': 1, 'fr': 1, 'es': 1, 'id': 1})
        self.assertRaises(LangDetectException, language, text=punctuation)
        self.assertRaises(LangDetectException, language, text=empty)

    def text_complexity(self):
        easy = "This is easy. This is a sentence. This has a big number."
        hard = "Quantum mechanics (QM; also known as quantum physics, quantum theory, the wave mechanical model, or matrix mechanics), including quantum field theory, is a fundamental theory in physics which describes nature at the smallest scales of energy levels of atoms and subatomic particles."
        punctuation = " . ,"
        empty = ""
        text_complexity = ComplexityMetadata().metadata_function
        self.assertEqual(text_complexity(empty), 0.0)
        self.assertEqual(text_complexity(punctuation), 0.0)
        self.assertEqual(text_complexity(easy), text_complexity(easy))
        self.assertGreater(text_complexity(hard), text_complexity(easy))
