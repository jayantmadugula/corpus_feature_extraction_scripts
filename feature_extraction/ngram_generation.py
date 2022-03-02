'''
These functions are used to create ngrams from
a generic input text dataset.
'''
from spacy.tokens.doc import Doc as sp_Doc

def generate_ngrams(doc: sp_Doc, n=2, pad_word='inv', idx_filter=None) -> list[str]:
    '''
    Generates a list of ngrams (or 'windows') from the given
    text with length 2`n` + 1.

    Returns a list of strings, each of length 2`n` + 1. Note, 
    each ngram is overlapping. If `text` has 5 words, then
    5 ngrams are generated.

    `doc` is expected to be a spaCy Doc. ngrams will be
    generated from this string.

    `pad_word` is the word used for padding.

    `pos_filter` can either be None or a list of indices indicating
    where words of interest are located in the text
    '''
    ngrams = []
    if idx_filter is not None:
        # pos_filter must be a list of indices
        iter_range = idx_filter
    else:
        # no part-of-speech filtering, iterate over every word
        iter_range = range(0, len(doc))

    for i in iter_range:
        ngram = generate_ngram_at_position(doc, i, n=n, pad_word=pad_word)
        ngrams.append(ngram)
    return ngrams

def generate_ngram_at_position(doc: sp_Doc, pos: int, n=2, pad_word='inv'):
    ''' 
    Generates an ngram with size 2`n` + 1 centered at the
    word at index `pos` in `doc`.
    '''
    w = doc[pos].text

    if pos - n < 0:
        len_padding = n - pos
        padding = [pad_word] * len_padding
        left_half = padding + [t.text for t in doc[0:pos]]
    else:
        left_half = [t.text for t in doc[pos-n:pos]]

    if pos + n + 1 > len(doc):
        len_padding = (pos + n + 1) - len(doc)
        padding = [pad_word] * len_padding
        right_half = [t.text for t in doc[pos+1:len(doc)]] + padding
    else:
        right_half = [t.text for t in doc[pos+1:pos+n+1]]

    ngram = ' '.join(left_half + [w] + right_half)
    return ngram