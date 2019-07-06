from collections import OrderedDict

delimiters = OrderedDict()
delimiters['HTML'] = r'<\s*br\s*/?\s*>|<\s*p\s*>'
delimiters['list'] = r'\n+\p{Pd}+'
delimiters['newline'] = r'\n+'
delimiters['sentence'] = r'\.+\s+|\!+\s+|\?+\s+'
delimiters['semicolon'] = r'\s*;\s'
delimiters['comma'] = r'\s*,\s'
delimiters['dash'] = r'\s*\p{Pd}+\s'
delimiters['tab'] = r'\t'
delimiters['whitespace'] = r'\s'


def most_common_n_to_string_frequency(histogram, n):
    if not isinstance(histogram, dict):
        return float('nan')
    sorted_histo = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    most_common_n = sorted_histo[:n]
    only_keys = [item[0] for item in most_common_n]
    return ', '.join(only_keys)


def most_common_n_to_string_alphabetically(histogram, n):
    if not isinstance(histogram, dict):
        return float('nan')
    sorted_histo = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    most_common_n = sorted_histo[:n]
    only_keys = [item[0] for item in most_common_n]
    sorted_keys = sorted(only_keys)
    return ', '.join(sorted_keys)
