'''
These functions are used to create ngrams from
a generic input text dataset.
'''
from spacy.tokens.doc import Doc as sp_Doc
from utilities.spacy_utilities import Spacy_Manager
import pandas as pd

def generate_corpus_ngrams(texts: pd.Series, n=2, pad_word='inv'):
    '''
    Manages ngram generation across a set of texts. These texts
    should be passed in as a `pd.Series` object.

    Returns a `pd.Series` of ngrams. Each entry in the Series is a ngram. The id of the corresponding sentence is included. 
    The number of these entries is equal to `len(texts)`.

    Return schema:
    - `ngram`
    - `sent_id`: the index of the sentence the ngram was 
    extracted from
    '''
    sp_docs = Spacy_Manager.generate_docs(texts)

    ngrams = []
    for d, sent_id in zip(sp_docs, texts.index):
        text_ngrams = generate_ngrams(d, n=n, pad_word=pad_word, idx_filter=None)
        sent_ids = [sent_id] * len(text_ngrams)
        ngrams.append(pd.DataFrame({'ngram': text_ngrams, 'sent_id': sent_ids}))
    
    return pd.concat(ngrams, ignore_index=True)

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

    `idx_filter` can either be None or a list of valid indices (for `doc`).
    
    If `idx_filter` is provided, only ngrams centered on words
    at the provided indices are created.
    '''
    ngrams = []
    if idx_filter is not None:
        # idx_filter must be a list of indices
        iter_range = idx_filter
    else:
        # no index filtering, iterate over every word
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