.. _text_metadata:

Text Metadata Precalculation
============================

Description
-----------

This precalculation computes metadata about a text. 
Since many checks cannot work with textual columns, this precalculation is used to numerize and categorize the texts.
It generates new columns with data about the textual column, such as:

NumCharsMetadata
    Returns the total number of characters in the text.

RatioUppercaseLettersMetadata
    Returns the number of uppercase characters divided by the number of alphabetical characters.

UnicodeCategoriesMetadata
    Returns a string, containing the unicode categories of the characters in the text, comma seperated, sorted descending by their frequency.
    What categories exist can be looked up here: https://www.fileformat.info/info/unicode/category/index.htm
    Example:
        input: 'Front-line leading edge website'
        
        output: 'Ll, Zs, Lu, Pd'

UnicodeBlocksMetadata
    Returns a string, containing the unicode blocks of the characters in the text, comma seperated, sorted descending by their frequency.
    What blocks exist can be looked up here: https://www.fileformat.info/info/unicode/block/index.htm
    Example:
        input: 'Front-line leading edge website'
        
        output: 'Basic Latin'

NumWordsMetadata
    Returns the total number of words in the text. A word is surrounded bz spaces or punctuation.
    Hyphens count as word delimiters, apostrophes do not count as delimiters.

DistinctWordsRatioMetadata
    Returns the number of distinct words devided by the total number of words.

UniqueWordsRatioMetadata
    Returns the number of hapax legomenon, words that only appear once in the text, devided by the total number of words.

UnknkownWordRatioMetadata
    Returns the number of words that are not in a dictionary devided by the total number of words.
    Popular reasons for words to be unknown: Typos, technical terms, neologisms.

StopwordRatioMetadata
    Returns the number of stopwords devided by the total number of words.
    Stopwords are defined by the nltk stopwords, supporting multiple languages. 

DelimiterTypeMetadata
    Returns a string, containing the delimiter type in the text.
    This are the possible delimiter types:
        'HTML' : text contains at least one html tag, such as break-tags <br/> or paragraph tags <p> in any variation.

        'newline' : text contains at least one '\\n' character and does not contain any of the above delimiters.

        'sentence' : text contains at least one of these punctuations: '.' '!' '?' and does not contain any of the above delimiters.

        'semicolon' : text contains at least one semicolon ';' and does not contain any of the above delimiters.

        'comma' : text contains at least one comma ',' and does not contain any of the above delimiters.

        'dash' : text contains at least one dash of any kind and does not contain any of the above delimiters.

        'tab' : text contains at least one '\\t' characters and does not contain any of the above delimiters.

        'whitespace' : text contains at least one whitespace '\\s' character and does not contain any of the above delimiters.

        'no delimiter' : text does not contain any of the above delimiters.


NumPartsMetadata
    Returns the number of parts in the text, the text is splitted into parts by it's delimiter type.

LanguagePerParagraph
    Returns a string, containing the languages of the parts in the text, comma seperated, sorted descending by their frequency.
    text is splitted into parts either by html <br/> tags or, if there are no html tags, by newlines.
    Language detection by  https://github.com/Mimino666/langdetect
    Example:
        input: 'Dieser Text ist zum Teil deutsch. \\n Part of this text is in english. \\n This is an example.'
        
        output: 'en, de'

LanguageMetadata
    Returns the language of the whole text.
    Language detection by https://github.com/Mimino666/langdetect

ComplexityMetadata
    Returns the complexity of the text. Using the text_standard text complexity from https://github.com/shivam5992/textstat
    The higher the number the more complex is the text. Only working for english language.

PartOfSpeechMetadata
    Returns a string, containing the part of speech tags in the text, comma seperated, sorted descending by their frequency.
    Uses the part of speech tagger by nltk. Only working for english language.
    Example:
        input: 'This is a normal sentence. This is for testing.'
        
        output: 'DET, VERB, ., ADJ, ADP, NOUN'
