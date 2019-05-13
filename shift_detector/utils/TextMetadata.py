import re
import numpy as np

from langdetect import DetectorFactory, detect, detect_langs
from iso639 import languages
import textstat
from spellchecker import SpellChecker
import unicodedata
import UCBlist
from nltk.corpus import stopwords

delimiter_HTML = '<\s*br\s*/?>|<\s*p\s*>'
delimiter_sentence = '\.\s'
delimiter_other = '\s*,\s|\s+-+\s+'
DetectorFactory.seed = 0


def md_functions(type):
    return {'num_chars': num_chars,
            'ratio_upper': ratio_upper,
            'unicode_category': unicode_category_histogram,
            'unicode_block': unicode_block_histogram,
            'num_words': num_words,
            'distinct_words': num_distinct_words,
            'unknown_words': unknown_word_ratio,
            'stopwords': stopword_ratio,
            'num_parts': num_parts,
            'category': category,
            'languages': language,
            'complexity': text_complexity}[type]

# preprocessors

def text_to_array(text):
    text = re.sub(r'[^\w\s]',' ',text)
    splitted = re.split('\W\s|\s', text)
    while '' in splitted:
        splitted.remove('')
    return splitted


def block(ch):
    # Return the Unicode block name for ch, or None if ch has no block.
    # from https://stackoverflow.com/questions/243831/unicode-block-of-a-character-in-python
    assert isinstance(ch, str) and len(ch) == 1, repr(ch)
    cp = ord(ch)
    for start, end, name in UCBlist._blocks:
        if start <= cp <= end:
        return name

# metrics
    
def num_distinct_words(text):
    distinct_words = []
    words = text_to_array(text)
    for word in words:
        if word not in distinct_words:
            distinct_words.append(word)
    return len(distinct_words)


def num_words(text):
    return len(text_to_array(text))


def min_num_words(data):
    return min([num_words(text) for text in data])


def num_chars(text):
    return len(text)


def unicode_category_histogram(text):
    characters = {}
    for c in text:
        category = unicodedata.category(c)
        if category in characters:
            characters[category] += 1
        else:
            characters[category] = 1
    return characters


def unicode_block_histogram(text):
    characters = {}
    for c in text:
        category = block(c)
        if category in characters:
            characters[category] += 1
        else:
            characters[category] = 1
    return characters


def ratio_upper(text):
    lower = sum(map(str.islower, text))
    upper = sum(1 for c in text if c.isupper())
    return round((upper * 100) / (lower + upper), 2)


def unknown_word_ratio(text):
    # not working for every language
    try:
        words = text_to_array(text)
        spell = SpellChecker(language=detect(text))

        misspelled = spell.unknown(words)
        return round(len(misspelled)*100 / len(words),2)
    except:
        return float('nan')


def category(text):
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


def num_parts(text):
    if (category(text) == 'html'):
        return len(re.split(delimiter_HTML, text))
    elif (category(text) == 'sentence'):
        return len(re.split(delimiter_sentence, text))
    elif (category(text) == 'other delimiter'):
        return len(re.split(delimiter_other, text))
    else:
        return 0


def language(text):
    parts = []
    if (category(text) == 'html'):
        parts = re.split(r'<\s*br\s*/?>', text)
    else:
        parts = re.split(r'[\n\r]+', text)
    parts = [x.strip() for x in parts if x.strip()]
    languages = {}
    for part in parts:
        lang = detect(part)
        if lang in languages:
            languages[lang] += 1
        else:
            languages[lang] = 1
    return languages


def text_complexity(text):
    # only working for english
    if(len(text) == 0):
        return 0
    complexity = float('nan')
    
    if (detect(text) == 'en'):
        complexity = textstat.flesch_reading_ease(text)
    return complexity


def stopword_ratio(text):
    # not working for every language
    try:
        stopword_count = 0
        words = text_to_array(text)
        language = detect(text)
        stop = stopwords.words(languages.get(part1=language).name.lower())
        for word in words:
            if word.lower() in stop:
                stopword_count += 1
        return round(stopword_count*100 / len(words),2)
    except: 
        return float('nan')