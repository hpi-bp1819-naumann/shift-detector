import sys
from IPython.display import display, Markdown


def nprint(input_str, num_tabs=0, text_formatting='normal'):

    for i in range(num_tabs):
        input_str = '\t' + input_str

    if not is_in_jupyter() or text_formatting == 'normal':
        print(input_str)

    if text_formatting == 'h1':
        display(Markdown('# {}'.format(input_str)))

    elif text_formatting == 'h2':
        display(Markdown('## {}'.format(input_str)))

    elif text_formatting == 'h3':
        display(Markdown('### {}'.format(input_str)))


# prints inline markdown
def mdprint(input_str):
    if not is_in_jupyter():
        print(input_str)
    else:
        display(Markdown(input_str))


def is_in_jupyter():
    return 'ipykernel' in sys.modules


def lprint(string, log_print):
    if log_print:
        print(string)
