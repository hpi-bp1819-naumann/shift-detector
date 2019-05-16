import re
import numpy as np

from langdetect import DetectorFactory, detect, detect_langs, lang_detect_exception
from iso639 import languages
import textstat
from spellchecker import SpellChecker
import unicodedata
import shift_detector.utils.UCBlist as UCBlist
from nltk.corpus import stopwords
import nltk

delimiter_HTML = r'<\s*br\s*/?\s*>|<\s*p\s*>'
delimiter_sentence = r'\.\s'
delimiter_other = r'\s*,\s|\s+-+\s+'
DetectorFactory.seed = 0


def md_functions(kind):
    return {'num_chars': num_chars,
            'ratio_upper': ratio_upper,
            'unicode_categories': unicode_category_string,
            'unicode_blocks': unicode_block_string,
            'num_words': num_words,
            'distinct_words': num_distinct_words,
            'unique_words': num_unique_words,
            'unknown_ratio': unknown_word_ratio,
            'stopword_ratio': stopword_ratio,
            'delimiter_type': delimiter_type,
            'num_parts': num_parts,
            'languages': language_string,
            'complexity': text_complexity}[kind]

# preprocessors


def text_to_array(text):
    text = re.sub(r'[^\w\s]','',text)
    splitted = re.split(r'\W\s|\s', text)
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

# postprocessors


def dictionary_to_sorted_string(histogram):
    sorted_histo = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    only_keys = [item[0] for item in sorted_histo]
    return ', '.join(only_keys)

def language_string(text):
    return dictionary_to_sorted_string(language(text))

def unicode_category_string(text):
    return dictionary_to_sorted_string(unicode_category_histogram(text))

def unicode_block_string(text):
    return dictionary_to_sorted_string(unicode_block_histogram(text))

# metrics


def num_chars(text):
    return len(text)
    
def ratio_upper(text):
    if text == "":
        return 0
    lower = sum(map(str.islower, text))
    upper = sum(1 for c in text if c.isupper())
    return round((upper * 100) / (lower + upper), 2)

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

def num_words(text):
    return len(text_to_array(text))

def num_distinct_words(text):
    distinct_words = []
    words = text_to_array(text)
    for word in words:
        if word not in distinct_words:
            distinct_words.append(word)
    return len(distinct_words)

def num_unique_words(text):
    words = text_to_array(text)
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

def unknown_word_ratio(text, language):
    # not working for every language
    try:
        words = text_to_array(text)
        spell = SpellChecker(language)

        if len(words) == 0:
            return 0.0

        misspelled = spell.unknown(words)
        return round(len(misspelled)*100 / len(words),2)
    except:
        pass

def stopword_ratio(text, language):
    # not working for every language
    try:
        stopword_count = 0
        words = text_to_array(text)
        stop = stopwords.words(languages.get(part1=language).name.lower())
        if(len(words) == 0):
            return 0.0
        for word in words:
            if word.lower() in stop:
                stopword_count += 1
        return round(stopword_count*100 / len(words),2)
    except: 
        pass

def delimiter_type(text):
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
    if (delimiter_type(text) == 'html'):
        return len(re.split(delimiter_HTML, text))
    elif (delimiter_type(text) == 'sentence'):
        return len(re.split(delimiter_sentence, text))
    elif (delimiter_type(text) == 'other delimiter'):
        return len(re.split(delimiter_other, text))
    else:
        return 0

def language(text):
    parts = []
    try: 
        if (len(text) == 0):
            detect(text) # trigger LangDetectException. Throwing one in here smh doesnt work
        if (delimiter_type(text) == 'html'):
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
    except:
        pass

def text_complexity(text):
    # lower value means more complex
    # works best for longer english texts. kinda works for other languages as well (not good though)
    complexity = textstat.textstat.flesch_reading_ease(text)
    return complexity

#def pos_histogram(text):
    