import unittest
import shift_detector.utils.TextMetadata as tm
from langdetect.lang_detect_exception import LangDetectException

class TestTextMetadata(unittest.TestCase):

    def test_text_to_array(self):
        normal = "This. is a'n example, ,, 12  35,6  , st/r--ing    \n test."
        empty = ""
        punctuation = ".  , * (  \n \t [}"
        self.assertEqual(tm.text_to_array(normal), ['This', 'is', 'an', 'example', '12', '356', 'string', 'test'])
        self.assertEqual(tm.text_to_array(empty), [])
        self.assertEqual(tm.text_to_array(punctuation), [])

    def test_num_chars(self):
        normal = "normaler Text"
        unicodes = "\u6667 is one char"
        punctuation = "., < \t \n !`"
        empty = ""
        self.assertEqual(tm.num_chars(normal), 13)
        self.assertEqual(tm.num_chars(unicodes), 13)
        self.assertEqual(tm.num_chars(punctuation), 11)
        self.assertEqual(tm.num_chars(empty), 0)

    def test_ratio_upper(self):
        lower = "no upper case letters"
        upper = "ALL UPPER CASE LETTERS"
        mixed1 = "FifTY fIFty"
        mixed2 = "Tre"
        empty = ""
        self.assertEqual(tm.ratio_upper(lower), 0.00)
        self.assertEqual(tm.ratio_upper(upper), 100.00)
        self.assertEqual(tm.ratio_upper(mixed1), 50.00)
        self.assertEqual(tm.ratio_upper(mixed2), 33.33)
        self.assertEqual(tm.ratio_upper(empty), 0.00)

    def test_unicode_category(self):
        lower = "justlowerletters"
        different = "\n \u0600 \uF8FF \uDB80 Hi \u02B7 \u0C99 \u1F8D \u094A \uA670 ∑ ´ 42 \u2169 ‚·°‡ﬁ›‹€⁄¡™£¢∞§¶•ªº‘«»æ…ÆÚ˘¯≤≤≥ ,;' "
        empty = ""
        self.assertEqual(tm.unicode_category_histogram(lower),{'Ll': 16})
        self.assertEqual(tm.unicode_category_histogram(different),{'Cc': 1, 'Zs': 16, 'Cf': 1, 'Co': 1, 'Cs': 1, 'Lu': 3, 'Ll': 3, 'Lm': 1, 'Lo': 3, 'Lt': 1, 'Mc': 1, 'Me': 1, 'Sm': 6, 'Sk': 3, 'Nd': 2, 'Nl': 1, 'Ps': 1, 'Po': 10, 'So': 2, 'Pf': 2, 'Pi': 3, 'Sc': 3})
        self.assertEqual(tm.unicode_category_histogram(empty), {})

    def test_unicode_block(self):
        latin = "Latin Letters! *with punctuation,!./ and numbers 983"
        different = "\n \u0600 \uF8FF \uDB80 Hi \u02B7 \u0C99 \u1F8D \u094A \uA670 ∑ ´ 42 \u2169 ‚·°‡ﬁ›‹€⁄¡™£¢∞§¶•ªº‘«»æ…ÆÚ˘¯≤≤≥ ,;' "
        empty = ""
        self.assertEqual(tm.unicode_block_histogram(latin),{'Basic Latin': 52})
        self.assertEqual(tm.unicode_block_histogram(different),{'Basic Latin': 24, 'Arabic': 1, 'Private Use Area': 1, 'High Private Use Surrogates': 1, 'Spacing Modifier Letters': 2, 'Kannada': 1, 'Greek Extended': 1, 'Devanagari': 1, 'Cyrillic Extended-B': 1, 'Mathematical Operators': 5, 'Latin-1 Supplement': 16, 'Number Forms': 1, 'General Punctuation': 8, 'Alphabetic Presentation Forms': 1, 'Currency Symbols': 1, 'Letterlike Symbols': 1})
        self.assertEqual(tm.unicode_block_histogram(empty), {})

    def test_num_words(self):
        distinct = "this are all different words"
        same = "same same, same. same"
        withPunctuation= "Punctuation. doesn't affect. the---y get deleted.   "
        empty = ""
        self.assertEqual(tm.num_words(distinct), 5)
        self.assertEqual(tm.num_words(same), 4)
        self.assertEqual(tm.num_words(withPunctuation), 6)
        self.assertEqual(tm.num_words(empty), 0)

    def test_num_distinct_words(self):
        distinct = "this are all different words"
        same = "same same, same. same"
        mixed = "there are doubled words and there are distinct words."
        capitalLetters = "Capital letters matter Matter"
        empty = ""
        self.assertEqual(tm.num_distinct_words(distinct), 5)
        self.assertEqual(tm.num_distinct_words(same), 1)
        self.assertEqual(tm.num_distinct_words(mixed), 6)
        self.assertEqual(tm.num_distinct_words(capitalLetters), 4)
        self.assertEqual(tm.num_distinct_words(empty), 0)

    def test_unknown_words(self):
        correctEnglish = "This is a correct sentence"
        incorrectEnglish = "Thiis is an incozyzyrrect sentence"
        unsupportedLanguage = "Aqoonyahanada caalamku waxay aad ugu murmaan sidii luuqadaha aduunku ku bilaabmeem." #This is somali
        punctuation = " . "
        empty = ""
        self.assertEqual(tm.unknown_word_ratio(correctEnglish, 'en'), 0.00)
        self.assertEqual(tm.unknown_word_ratio(incorrectEnglish, 'en'), 40.00)
        self.assertRaises(ValueError, tm.unknown_word_ratio(unsupportedLanguage, 'so'))
        self.assertEqual(tm.unknown_word_ratio(punctuation, 'en'), 00.00)
        self.assertEqual(tm.unknown_word_ratio(empty, 'en'), 00.00)

    def test_stopwords(self):
        noStopwords = "computer calculates math"
        onlyStopwords = "The and is I am"
        mixed = "A normal sentence has both"
        unsupportedLanguage = "Aqoonyahanada caalamku waxay aad ugu murmaan sidii luuqadaha aduunku ku bilaabmeem."
        punctuation = " . "
        empty = ""
        self.assertEqual(tm.stopword_ratio(noStopwords, 'en'), 0.0)
        self.assertEqual(tm.stopword_ratio(onlyStopwords, 'en'), 100.0)
        self.assertEqual(tm.stopword_ratio(mixed, 'en'), 60.0)
        self.assertRaises(ValueError, tm.stopword_ratio(unsupportedLanguage, 'so'))
        self.assertEqual(tm.stopword_ratio(punctuation, 'en'), 0.0)
        self.assertEqual(tm.stopword_ratio(empty, 'en'), 0.0)
        
    def test_num_parts(self):
        html = "some text <p> some other text < br/ > more text"
        sentence = "some text. some other text. more text."
        other = "some text, some other text -- more text."
        none = "some text some other text more text"
        htmlsentence = "some text <p> some other text. more text."
        sentenceother = "some text, some other text. more text."
        htmlsentenceother = "some text -- some. other text <br> more text."
        empty = ""
        self.assertEqual(tm.num_parts(html), 3)
        self.assertEqual(tm.num_parts(sentence), 3)
        self.assertEqual(tm.num_parts(other), 3)
        self.assertEqual(tm.num_parts(none), 0)
        self.assertEqual(tm.num_parts(htmlsentence), 2)
        self.assertEqual(tm.num_parts(sentenceother), 2)
        self.assertEqual(tm.num_parts(htmlsentenceother), 2)
        self.assertEqual(tm.num_parts(empty), 0)

    def test_category(self):
        html = "some text <p> some other text < br/ > more text"
        sentence = "some text. some other text. more text."
        other = "some text, some other text -- more text."
        none = "some text some other text more text"
        htmlsentence = "some text <p> some other text. more text."
        sentenceother = "some text, some other text. more text."
        htmlsentenceother = "some text -- some. other text <br> more text."
        empty = ""
        self.assertEqual(tm.category(html), "html")
        self.assertEqual(tm.category(sentence), "sentence")
        self.assertEqual(tm.category(other), "other delimiter")
        self.assertEqual(tm.category(none), "no delimiter")
        self.assertEqual(tm.category(htmlsentence), "html")
        self.assertEqual(tm.category(sentenceother), "sentence")
        self.assertEqual(tm.category(htmlsentenceother), "html")
        self.assertEqual(tm.category(empty), "empty")

    def test_languages(self):
        english = "This is a normal sentence. Language detection is easy."
        englishTypos = "Thhis is a nirnal sentense. Lanquage detecction is esay."
        german = "Dies ist ein einfacher Satz. Kurz und knackig."
        englishgerman = "Dieser Text ist zum Teil deutsch. <br> Part of this text is in english"
        multipleLanguages = "Dieser Text ist zum Teil deutsch. \n Part of this text is in english. \n there actually is some french coming. \n Ce n'est pas anglais. \n No puedo hablar español. \n Beberapa bahasa untuk diuji."
        punctuation = " . ,"
        empty = ""
        self.assertEqual(tm.language(english), {'en': 1})
        self.assertEqual(tm.language(englishTypos), {'en': 1})
        self.assertEqual(tm.language(german), {'de': 1})
        self.assertEqual(tm.language(englishgerman), {'en': 1, 'de': 1})
        self.assertEqual(tm.language(multipleLanguages), {'en': 2, 'de': 1, 'fr': 1, 'es': 1, 'id': 1})
        self.assertRaises(LangDetectException, tm.language(punctuation))
        self.assertRaises(LangDetectException, tm.language(empty))

    def text_complexity(self):
        easy = "This is easy. This is a sentence. This has a big number."
        hard = "Quantum mechanics (QM; also known as quantum physics, quantum theory, the wave mechanical model, or matrix mechanics), including quantum field theory, is a fundamental theory in physics which describes nature at the smallest scales of energy levels of atoms and subatomic particles."
        punctuation = " . ,"
        empty = ""
        self.assertEqual(tm.text_complexity(empty), 206.84)
        self.assertEqual(tm.text_complexity(punctuation), 206.84)
        self.assertEqual(tm.text_complexity(easy), tm.text_complexity(easy))
        self.assertGreater(tm.text_complexity(easy), tm.text_complexity(hard))
