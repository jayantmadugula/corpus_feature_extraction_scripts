import unittest
from processing_functions import text_preprocessing
import pandas as pd

class TextPreprocessingTests(unittest.TestCase):
    def test_lowercase(self):
        test_df = pd.Series([
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        ])
        result = text_preprocessing.lowercase_words(test_df)
        expected = pd.Series([
            "lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        ])
        assert((result == expected).all())

    def test_remove_punctuation(self):
        test_df = pd.Series([
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        ])
        result = text_preprocessing.remove_punctuation(test_df)
        expected = pd.Series([
            "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
            "Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur"
        ])
        assert((result == expected).all())

    def test_normalized_spacing(self):
        test_df = pd.Series([
            "Lorem ipsum dolor sit amet, consectetur    adipiscing elit, sed  do eiusmod tempor incididunt   ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis   nostrud exercitation ullamco laboris    nisi ut aliquip ex ea commodo consequat.",
            "Duis aute   irure dolor in    reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        ])
        result = text_preprocessing.normalize_spacing(test_df)
        expected = pd.Series([
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        ])
        assert((result == expected).all())

    def test_remove_stopwords(self):
        test_df = pd.Series([
            "This is a random test sentence and it contains some stopwords.",
            "Here's another random test sentence, it also includes a few stopwords."
        ])
        result = text_preprocessing.remove_stopwords(test_df)
        expected = pd.Series([
            "This random test sentence contains stopwords.",
            "Here's another random test sentence, also includes stopwords."
        ])
        assert((result == expected).all())