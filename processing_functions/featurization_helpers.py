'''
This file contains a series of functions used to help with
feature extraction and ngram creation.
'''
from typing import Iterable
from spacy.tokens.doc import Doc as sp_Doc

# Part-of-Speech functions
def generate_pos_tags(docs: Iterable[sp_Doc], is_ngrams=True):
    '''
    Returns part-of-speech tags for word in the center
    of each text in `docs` by default.
    `docs` must be an iterable of spaCy Docs.
    
    If `is_ngrams` is set to False, then we generate a 
    part-of-speech tag for every word in each text.

    Note: For SemEval analysis, this function should be called
    on ngrams.
    '''
    if is_ngrams:
        word_pos_tags = _generate_pos_tags_ngram(docs)
    else:
        word_pos_tags = _generate_pos_tags(docs)
    return word_pos_tags

def _generate_pos_tags_ngram(docs: Iterable[sp_Doc], return_hash=False):
    ''' 
    Returns tag of middle word for each document. 

    When `return_hash` is `True`, the spaCy hash associated
    with the part-of-speech is returned instead of the string.
    '''
    for d in docs:
        if return_hash:
            doc_tag = d[int(len(d)/2)].pos
        else:
            doc_tag = d[int(len(d)/2)].pos_
        yield doc_tag

def _generate_pos_tags(docs: Iterable[sp_Doc], return_hash=False):
    ''' 
    Returns a tag for every word in each document. 

    When `return_hash` is `True`, the spaCy hash associated
    with the part-of-speech is returned instead of the string.
    '''
    for d in docs:
        if return_hash:
            doc_tags = (w.pos for w in d)
        else:
            doc_tags = (w.pos_ for w in d)
        yield doc_tags