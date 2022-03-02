'''
These functions apply generic text preprocessing functions
to a pandas Series.
'''

from string import punctuation
import pandas as pd


def lowercase_words(texts: pd.Series) -> pd.Series:
    return texts.str.lower()

def remove_punctuation(texts: pd.Series) -> pd.Series:
    return texts.str.translate(str.maketrans('', '', punctuation))

def normalize_spacing(texts: pd.Series) -> pd.Series:
    return texts.str.split().str.join(' ')