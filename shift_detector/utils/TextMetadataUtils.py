import re

from langdetect import DetectorFactory
from collections import OrderedDict

delimiters = OrderedDict()
delimiters['HTML'] = r'<\s*br\s*/?\s*>|<\s*p\s*>'
delimiters['list'] = r'\n+\p{Pd}+'
delimiters['newline'] = r'\n+'
delimiters['sentence'] = r'(\.|\!|\?)\s'
delimiters['semicolon'] = r'\s*;\s'
delimiters['comma'] = r'\s*,\s'
delimiters['dash'] = r'\s*\p{Pd}+\s'
delimiters['tab'] = r'\t'
delimiters['whitespace'] = r'\s'

DetectorFactory.seed = 0


def tokenize(text):
    text = re.sub(r'[^\w\s]','',text)
    splitted = re.split(r'\W\s|\s', text)
    while '' in splitted:
        splitted.remove('')
    return splitted


def dictionary_to_sorted_string(histogram):
    sorted_histo = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    only_keys = [item[0] for item in sorted_histo]
    return ', '.join(only_keys)
