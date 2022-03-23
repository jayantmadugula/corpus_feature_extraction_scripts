import unittest
from processing_functions import ngram_generation
from utilities.spacy_utilities import Spacy_Manager
import pandas as pd

class NgramGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        # test_strings from https://en.wikipedia.org/wiki/Lorem_ipsum
        self.test_strings = [
            "In publishing and graphic design Lorem ipsum is a placeholder text commonly used to demonstrate the visual form of a document or a typeface without relying on meaningful content",
            "Lorem ipsum may be used as a placeholder before the final copy is available",
            "It is also used to temporarily replace text in a process called greeking which allows designers to consider the form of a webpage or publication without the meaning of the text influencing the design"
        ]
        self.test_docs = list(Spacy_Manager.generate_docs(self.test_strings))
        return super().setUp()

    def test_generate_corpus_ngrams(self):
        result = ngram_generation.generate_corpus_ngrams(pd.DataFrame({'test': self.test_docs}), 'test')
        assert(result.shape[0] == sum([len(x.split()) for x in self.test_strings]))
        assert(result.shape[1] == 2)
        assert(result.index[0] == 0 and result.index[-1] == sum([len(x.split()) for x in self.test_strings]) - 1)

    def test_ngram_generation_at_position_no_padding(self):
        result = ngram_generation.generate_ngram_at_position(self.test_docs[0], 3)
        expected = "publishing and graphic design Lorem"
        assert(result == expected)
    
    def test_ngram_generation_at_position_with_padding(self):
        result = ngram_generation.generate_ngram_at_position(self.test_docs[0], 1)
        expected = "inv In publishing and graphic"
        assert(result == expected)

    def test_doc_ngram_generation(self):
        result = ngram_generation.generate_ngrams(self.test_docs[0])
        
        expected_ngrams = {
            1: "inv In publishing and graphic",
            3: "publishing and graphic design Lorem",
            -1: "on meaningful content inv inv"
        }
        
        assert(len(result) == len(self.test_strings[0].split()))
        for i, s in expected_ngrams.items():
            assert(result[i] == s)
