import unittest
from feature_extraction import ngram_generation
from utilities.spacy_utilities import Spacy_Manager

class NgramGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_strings = [
            "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
            "Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur"
        ]
        self.test_docs = list(Spacy_Manager.generate_docs(self.test_strings))
        return super().setUp()

    def test_ngram_generation_at_position_no_padding(self):
        result = ngram_generation.generate_ngram_at_position(self.test_docs[0], 3)
        expected = "ipsum dolor sit amet consectetur"
        assert(result == expected)
    
    def test_ngram_generation_at_position_with_padding(self):
        result = ngram_generation.generate_ngram_at_position(self.test_docs[0], 1)
        expected = "inv Lorem ipsum dolor sit"
        assert(result == expected)

    def test_doc_ngram_generation(self):
        result = ngram_generation.generate_ngrams(self.test_docs[0])
        
        expected_ngrams = {
            1: "inv Lorem ipsum dolor sit",
            3: "ipsum dolor sit amet consectetur",
            -1: "dolore magna aliqua inv inv"
        }
        
        assert(len(result) == len(self.test_strings[0].split()))
        for i, s in expected_ngrams.items():
            assert(result[i] == s)
