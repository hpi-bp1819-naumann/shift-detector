import re
from langdetect import DetectorFactory, detect, detect_langs, lang_detect_exception
from iso639 import languages
import textstat
from spellchecker import SpellChecker
import unicodedata
import shift_detector.utils.UCBlist as UCBlist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import treetaggerwrapper
import string
from collections import defaultdict

delimiter_HTML = r'<\s*br\s*/?\s*>|<\s*p\s*>'
delimiter_sentence = r'\.\s'
delimiter_other = r'\s*,\s|\s+-+\s+'
DetectorFactory.seed = 0


def md_functions(type):
    return {'dict_to_string': dictionary_to_sorted_string,
            'num_chars': num_chars,
            'ratio_upper': ratio_upper,
            'unicode_category': unicode_category_histogram,
            'unicode_block': unicode_block_histogram,
            'num_words': num_words,
            'distinct_words': num_distinct_words,
            'unique_words': num_unique_words,
            'unknown_words': unknown_word_ratio,
            'stopwords': stopword_ratio,
            'category': category,
            'num_parts': num_parts,
            'languages': language_dict,
            'language': language,
            'complexity': text_complexity}[type]

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

# metrics

def num_chars(text):
    return len(text)
    
def ratio_upper(text):
    lower = sum(map(str.islower, text))
    upper = sum(1 for c in text if c.isupper())
    if (lower + upper) == 0:
        return 0
    return round((upper * 100) / (lower + upper), 2)

def unicode_category_histogram(text):
    characters = defaultdict(int)
    for c in text:
        category = unicodedata.category(c)
        characters[category] += 1
    return characters

def unicode_block_histogram(text):
    characters = defaultdict(int)
    for c in text:
        category = block(c)
        characters[category] += 1
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

def distinct_words_ratio(text):
    if (num_words(text) == 0):
        return 0
    return round(num_distinct_words(text)*100/num_words(text),2)

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

def unique_words_ratio(text):
    if (num_words(text) == 0):
        return 0
    return round(num_unique_words(text)*100/num_words(text),2)

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

def language_dict(text):
    parts = []
    try: 
        if (len(text) == 0):
            detect(text) # trigger LangDetectException. Throwing one in here smh doesnt work
        if (category(text) == 'html'):
            parts = re.split(r'<\s*br\s*/?>', text)
        else:
            parts = re.split(r'[\n\r]+', text)
        parts = [x.strip() for x in parts if x.strip()]
        languages = defaultdict(int)
        for part in parts:
            lang = detect(part)
            languages[lang] += 1
        return languages
    except:
        pass

def language(text):
    return detect(text)

def complexity(text):
    # works best for longer english texts. kinda works for other languages as well (not good though)
    complexity = textstat.textstat.text_standard(text, True)
    return complexity

def pos_histogram(text, language):
    try:
        tagger = treetaggerwrapper.TreeTagger(TAGLANG=language)
        tagged = treetaggerwrapper.make_tags(tagger.tag_text(text))
        tags = defaultdict(int)
        for item in tagged:
            if not isinstance(item,treetaggerwrapper.Tag): #Tag as UNK for unknown
                tags['UNK'] += 1
            else:
                tags[item.pos] += 1
        return tags
    except:
        pass
  
def smaller_pos_histogram(pos_histogram):
    smaller = defaultdict(int)
    for key, value in pos_histogram.items():
        if all(j in string.punctuation for j in key):
            smaller['SEN'] += value
        else:
            smaller[key.split(':')[0][:3]] += value
    return smaller

def pos_ratio_histogram(pos_histogram):
    possum = sum(pos_histogram.values())
    pos_ratios = {}
    for key, value in pos_histogram.items():
        pos_ratios[key] = round(value*100/possum,2)
    return pos_ratios
  