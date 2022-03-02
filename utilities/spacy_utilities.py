'''
This file contains utilities for spaCy.
'''

import spacy
import numpy as np

class Spacy_Manager:
    _nlp = spacy.load('en_core_web_lg')
    def __init__(self):
        return

    @classmethod
    def generate_docs(cls, texts, batch_size=1000, n_threads=2):
        return cls._nlp.pipe(texts, batch_size=batch_size, n_process=n_threads)

def get_doc_vectors(docs):
    ''' Returns word vectors for all texts in `docs` using spaCy '''
    return np.stack([d.vector for d in docs])

def get_doc_vector(doc):
    ''' spaCy `doc` returns mean of word vectors in `doc` '''
    return doc.vector

def get_doc_tokens(docs):
    tags = []
    for d in docs:
        doc_tags = [t.tag for t in d]
        tags.append(doc_tags)
    return np.stack(tags)

'''
Links:
https://stackoverflow.com/questions/53118666/spacy-convert-token-type-into-list
'''