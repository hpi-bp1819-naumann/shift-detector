import re
import numpy as np

from langdetect import DetectorFactory, detect, detect_langs

delimiterHTML = '<\s*br\s*/?>|<\s*p\s*>'
delimiterSentence = '\.\s'
delimiterOther = '\s*,\s|\s+-+\s+'
DetectorFactory.seed = 0


def md_functions(type):
    return {'num_chars': num_chars,
            'ratio_upper': ratio_upper,
            'num_words': num_words,
            'distinct_words': num_distinct_words,
            'num_parts': num_parts,
            'category': category,
            'language': language,
            'lang_ambiguity': lang_count}[type]


def num_distinct_words(text):
    distinct_words = []
    text = re.sub(r'[^\w\s]', ' ', text)
    while "  " in text:
        text = text.replace("  ", " ")
    words = re.split('\W\s|\s', text)
    while '' in words:
        words.remove('')
    for word in words:
        if word not in distinct_words:
            distinct_words.append(word)
    return len(distinct_words)


def num_words(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    while "  " in text:
        text = text.replace("  ", " ")
    splitted = re.split('\W\s|\s', text)
    while '' in splitted:
        splitted.remove('')
    return len(splitted)


def min_num_words(data):
    return min([num_words(text) for text in data])


def num_chars(text):
    return len(text)


def ratio_upper(text):
    lower = sum(map(str.islower, text))
    upper = sum(1 for c in text if c.isupper())
    return round((upper * 100) / (lower + upper), 2)


def category(text):
    html = re.compile('<.*?>')
    point = re.compile(delimiterSentence)
    other = re.compile(delimiterOther)
    if (html.search(text)):
        return 'html'
    elif (point.search(text)):
        return 'sentence'
    elif (other.search(text)):
        return 'otherDelimiter'
    elif (len(text) == 0):
        return 'empty'
    else:
        return 'noDelimiter'


def num_parts(text):
    if (category(text) == 'html'):
        return len(re.split(delimiterHTML, text))
    elif (category(text) == 'sentence'):
        return len(re.split(delimiterSentence, text))
    elif (category(text) == 'otherDelimiter'):
        return len(re.split(delimiterOther, text))
    else:
        return 0


def language(text):
    l = 'unknown'
    try:
        l = detect(text)
    except:
        l = 'unknown'
    return l


def languages(text):
    l = 'unknown'
    try:
        l = ', '.join([language.lang for language in detect_langs(text)])
    except:
        l = 'unknown'
    return l


def lang_count(text):
    c = np.nan
    try:
        c = len(detect_langs(text))
    except:
        c = np.nan
    return c
