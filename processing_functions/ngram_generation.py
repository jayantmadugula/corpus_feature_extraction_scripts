'''
These functions are used to create ngrams from
a generic input text dataset.
'''
from spacy.tokens.doc import Doc as sp_Doc
import pandas as pd
from processing_functions.featurization_helpers import generate_pos_tags


def generate_corpus_ngrams(input_df: pd.DataFrame, col_name: str, n=2, pad_word='inv', **kwargs):
    '''
    Manages ngram generation across a set of texts. These texts
    should be passed in as a `pd.Series` object.

    Returns a `pd.Series` of ngrams. Each entry in the Series is a ngram. 
    The id of the corresponding sentence is included. 
    The number of these entries is equal to `len(texts)`.

    `kwargs` supports the following arguments:
    1. `"pos_filter"`, which must be a list of valid parts-of-speech from spaCy,
    will limit ngram creation to ngrams where the central "target" word has a
    part-of-speech included in the provided list.
    2. `"idx_filter"`, which must be an iterable containing valid indices, limits ngram
    creation to the provided indices for each document.
    3. `"include_metadata"`, can either be a boolean or a list of column names. If
    set to True, all columns, except `col_name` in `input_df` are joined to the returned
    DataFrame. If a list of column names are provided, then only those columns are
    joined with the returned DataFrame.

    Return schema:
    - `ngram`
    - `sent_id`: the index of the sentence the ngram was 
    extracted from
    '''
    sp_docs = input_df.loc[:, col_name]
    
    if 'pos_filter' in kwargs:
        # Create part-of-speech filter and get indices at which the filter is valid.
        pos_tags = generate_pos_tags(sp_docs, is_ngrams=False)
        pos_idx_filter = _create_tag_filter(pos_tags, set(kwargs['pos_filter']))
        zipped_ngram_iterator = zip(sp_docs, input_df.index, pos_idx_filter)
    elif 'idx_filter' in kwargs:
        # Simply use the existing index-based filter.
        zipped_ngram_iterator = zip(sp_docs, input_df.index, kwargs['idx_filter'])
    else:
        zipped_ngram_iterator = zip(sp_docs, input_df.index)

    # Calculate ngrams at valid indices.
    ngrams = []
    for i in zipped_ngram_iterator:
        if len(i) == 3:
            d, sent_id, idx_filter = i
        else:
            d, sent_id = i
            idx_filter = None

        text_ngrams = generate_ngrams(d, n=n, pad_word=pad_word, idx_filter=idx_filter)
        sent_ids = [sent_id] * len(text_ngrams)
        ngrams.append(pd.DataFrame({'ngram': text_ngrams, 'sent_id': sent_ids}))
    
    ngrams_df: pd.DataFrame = pd.concat(ngrams, ignore_index=True)
    if 'include_metadata' in kwargs:
        if type(kwargs['include_metadata']) == list:
            metadata_cols = kwargs['include_metadata']
            return ngrams_df.join(input_df.loc[:, metadata_cols], on='sent_id', how='inner')
        elif kwargs['include_metadata'] == True:
            metadata_cols = set(input_df.columns) - {col_name}
            return ngrams_df.join(input_df.loc[:, metadata_cols], on='sent_id', how='inner')
        elif kwargs['include_metadata'] == False:
            return ngrams_df
        else:
            raise ValueError('The "include_metadata" parameter must be a list or boolean.')
    
    return ngrams_df

def _create_tag_filter(tags, tag_filter):
    '''
    Returns a list of valid indices given a tag-based filter.
    '''
    texts_idx = []
    for text_tags in tags:
        idx = []
        for i, pos in enumerate(text_tags):
            if pos in tag_filter: idx.append(i)
        texts_idx.append(idx)
    return texts_idx

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