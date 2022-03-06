'''
These functions apply generic text preprocessing functions
to a pandas Series.
'''

from string import punctuation
import pandas as pd
from nltk.corpus import stopwords


def lowercase_words(texts: pd.Series) -> pd.Series:
    return texts.str.lower()

def remove_punctuation(texts: pd.Series) -> pd.Series:
    return texts.str.translate(str.maketrans('', '', punctuation))

def normalize_spacing(texts: pd.Series) -> pd.Series:
    return texts.str.split().str.join(' ')

def remove_stopwords(texts: pd.Series) -> pd.Series:
    '''
    Removes stopwords from text using the NLTK English stopwords list.
    `texts` must be an pandas Series object.
    Returns a pandas Series object, with all stopwords removed.
    '''
    split_text = texts.str.split()
    stop = set(stopwords.words('english'))

    split_text = split_text.apply(lambda x: [w for w in x if w not in stop])

    return split_text.str.join(' ')