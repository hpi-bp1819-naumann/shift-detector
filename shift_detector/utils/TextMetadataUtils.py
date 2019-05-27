import re

from langdetect import DetectorFactory

delimiter_HTML = r'<\s*br\s*/?\s*>|<\s*p\s*>'
delimiter_sentence = r'\.\s'
delimiter_other = r'\s*,\s|\s+-+\s+'
DetectorFactory.seed = 0


def text_to_array(text):
    text = re.sub(r'[^\w\s]','',text)
    splitted = re.split(r'\W\s|\s', text)
    while '' in splitted:
        splitted.remove('')
    return splitted


def dictionary_to_sorted_string(histogram):
    sorted_histo = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    only_keys = [item[0] for item in sorted_histo]
    return ', '.join(only_keys)
