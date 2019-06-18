import sys
from IPython.display import display, Markdown


def nprint(input_str, num_tabs=0, text_formatting='normal'):

    for i in range(num_tabs):
        input_str = '\t' + input_str

    if not is_in_jupyter() or text_formatting == 'normal':
        print(input_str)

    if text_formatting == 'h1':
        display(Markdown('# {}'.format(input_str)))

    if text_formatting == 'h2':
        display(Markdown('## {}'.format(input_str)))


def is_in_jupyter():
    return 'ipykernel' in sys.modules
