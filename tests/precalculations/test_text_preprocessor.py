import unittest

from pandas.util.testing import assert_frame_equal

from shift_detector.precalculations.Store import Store
from shift_detector.precalculations.TextPreprocessor import *


class TestTextPreprocessorFunctions(unittest.TestCase):

    def setUp(self):
        poems = [
            'Tell me not, in mournful numbers,\n'
            'Life is but an empty dream!\n'
            'For the soul is dead that slumbers,\n'
            'And things are not what they seem.',
            'Life is real! Life is earnest!\n'
            'And the grave is not its goal;\n'
            'Dust thou art, to dust returnest,\n'
            'Was not spoken of the soul.',
            'Not enjoyment, and not sorrow,\n'
            'Is our destined end or way;\n'
            'But to act, that each tomorrow\n'
            'Find us farther than today.',
            'So, gilded by the glow of youth,\n'
            'Our varied life looks fair and gay;\n'
            'And so remains the naked truth,\n'
            'When that false light is past away.',
            'Fond dreamer! little does she know\n'
            'The anxious toil, the suffering,\n'
            'The blasted hopes, the burning woe,\n'
            'The object of her joy will bring.',
            'Trust no Future, howe’er pleasant!\n'
            'Let the dead Past bury its dead!\n'
            'Act,— act in the living Present!\n'
            'Heart within, and God o’erhead! ',
            'They do not see how cruel Death\n'
            'Comes on, their loving hearts to part:\n'
            'One feels not now the gasping breath,\n'
            'The rending of the earth-bound heart, --',
            "Rapidly, merrily,\n"
            "Life's sunny hours flit by,\n"
            "Gratefully, cheerily,\n"
            "Enjoy them as they fly !",
            "It has neither a beginning nor an end\n"
            "You can never predict where it will bend.",
            'Life is a teacher, it will show you the way\n'
            'But unless you live it...it will run away.']
        phrases = ['Front-line leading edge website',
                   'Upgradable upward-trending software',
                   'Virtual tangible throughput',
                   'Robust secondary open system',
                   'Devolved multimedia knowledge user',
                   'Self-enabling next generation capability',
                   'Automated 3rd generation benchmark',
                   'Switchable global info-mediaries',
                   'Automated 24/7 alliance',
                   'Robust logistical function']
        df1 = pd.DataFrame.from_dict({'text': poems})
        df2 = pd.DataFrame.from_dict({'text': phrases})
        self.store = Store(df1, df2)

    def test_tokenize_into_words_preprocessor(self):
        md1, md2 = TokenizeIntoLowerWordsPrecalculation().process(self.store)
        sol1_words = [
            [['tell', 'me', 'not', 'in', 'mournful', 'numbers', 'life', 'is', 'but', 'an', 'empty', 'dream', 'for',
              'the', 'soul', 'is', 'dead', 'that', 'slumbers', 'and', 'things', 'are', 'not', 'what', 'they', 'seem']],
            [['life', 'is', 'real', 'life', 'is', 'earnest', 'and', 'the', 'grave', 'is', 'not', 'its', 'goal', 'dust',
              'thou', 'art', 'to', 'dust', 'returnest', 'was', 'not', 'spoken', 'of', 'the', 'soul']],
            [['not', 'enjoyment', 'and', 'not', 'sorrow', 'is', 'our', 'destined', 'end', 'or', 'way', 'but', 'to',
              'act', 'that', 'each', 'tomorrow', 'find', 'us', 'farther', 'than', 'today']],
            [['so', 'gilded', 'by', 'the', 'glow', 'of', 'youth', 'our', 'varied', 'life', 'looks', 'fair',
              'and', 'gay', 'and', 'so', 'remains', 'the', 'naked', 'truth', 'when', 'that', 'false', 'light',
              'is', 'past', 'away']],
            [['fond', 'dreamer', 'little', 'does', 'she', 'know', 'the', 'anxious', 'toil', 'the',
              'suffering', 'the', 'blasted', 'hopes', 'the', 'burning', 'woe', 'the', 'object', 'of', 'her',
              'joy', 'will', 'bring']],
            [['trust', 'no', 'future', 'howeer', 'pleasant', 'let', 'the', 'dead', 'past', 'bury', 'its',
              'dead', 'act', 'act', 'in', 'the', 'living', 'present', 'heart', 'within', 'and', 'god',
              'oerhead']],
            [['they', 'do', 'not', 'see', 'how', 'cruel', 'death', 'comes', 'on', 'their', 'loving', 'hearts',
              'to', 'part', 'one', 'feels', 'not', 'now', 'the', 'gasping', 'breath', 'the', 'rending', 'of',
              'the', 'earth', 'bound', 'heart']],
            [['rapidly', 'merrily', "life's", 'sunny', 'hours', 'flit', 'by', 'gratefully', 'cheerily',
              'enjoy', 'them', 'as', 'they', 'fly']],
            [['it', 'has', 'neither', 'a', 'beginning', 'nor', 'an', 'end', 'you', 'can', 'never', 'predict',
              'where', 'it', 'will', 'bend']],
            [['life', 'is', 'a', 'teacher', 'it', 'will', 'show', 'you', 'the', 'way', 'but', 'unless', 'you',
              'live', 'itit', 'will', 'run', 'away']]
        ]
        sol2_words = [
            [['front', 'line', 'leading', 'edge', 'website']],
            [['upgradable', 'upward', 'trending', 'software']],
            [['virtual', 'tangible', 'throughput']],
            [['robust', 'secondary', 'open', 'system']],
            [['devolved', 'multimedia', 'knowledge', 'user']],
            [['self', 'enabling', 'next', 'generation', 'capability']],
            [['automated', '3rd', 'generation', 'benchmark']],
            [['switchable', 'global', 'info', 'mediaries']],
            [['automated', '247', 'alliance']],
            [['robust', 'logistical', 'function']]
        ]
        solution1 = pd.DataFrame(sol1_words, columns=['text'])
        solution2 = pd.DataFrame(sol2_words, columns=['text'])
        assert_frame_equal(solution1, md1)
        assert_frame_equal(solution2, md2)

    def test_tokenize_into_words_function(self):
        normal = "This. is a'n example, ,, 12  35,6  , st/r--ing    \n test."
        empty = ""
        punctuation = ".  , * (  \n \t [}"
        tokenize_into_words = TokenizeIntoLowerWordsPrecalculation().tokenize_into_words
        self.assertEqual(tokenize_into_words(normal),
                         ['this', 'is', "a'n", 'example', '12', '356', 'str', 'ing', 'test'])
        self.assertEqual(tokenize_into_words(empty), [])
        self.assertEqual(tokenize_into_words(punctuation), [])
