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

        self.test_metadata = [
            1,
            0,
            -1
        ]

        return super().setUp()

    def test_generate_corpus_ngrams(self):
        test_col_name = 'test'
        test_df = pd.DataFrame({test_col_name: self.test_docs})

        # Test without filtering.
        result = ngram_generation.generate_corpus_ngrams(test_df, test_col_name)
        assert(result.shape[0] == sum([len(x.split()) for x in self.test_strings]))
        assert(result.shape[1] == 2)
        assert(result.index[0] == 0 and result.index[-1] == sum([len(x.split()) for x in self.test_strings]) - 1)

        # Test with filtering.
        test_idx_filter = [[0, 3, 4, 6, 7, 10, 13], [3, 4, 8], [0, 1, 2, 9, 11]]
        idx_filter_result = ngram_generation.generate_corpus_ngrams(test_df, test_col_name, idx_filter=test_idx_filter)
        assert(idx_filter_result.shape[0] == sum(len(x) for x in test_idx_filter))
        assert(idx_filter_result.shape[1] == 2)
        assert(idx_filter_result.index[0] == 0 and idx_filter_result.index[-1] == sum(len(x) for x in test_idx_filter) - 1)

        # Test with PoS filtering.
        test_pos_filter = ['NOUN', 'ADV', 'PRON']
        pos_filter_result = ngram_generation.generate_corpus_ngrams(test_df, test_col_name, pos_filter=test_pos_filter)
        assert(pos_filter_result.shape[0] <= sum([len(x.split()) for x in self.test_strings]))
        assert(pos_filter_result.shape[1] == 2)
        assert(pos_filter_result.index[0] == 0 and pos_filter_result.index[-1] <= sum([len(x.split()) for x in self.test_strings]) - 1)

    def test_generate_corpus_ngrams_with_metadata(self):
        test_col_name = 'test'
        metadata_col_name = 'metadata_col'
        test_df = pd.DataFrame({test_col_name: self.test_docs, metadata_col_name: self.test_metadata})

        # Test with boolean parameter.
        result = ngram_generation.generate_corpus_ngrams(test_df, test_col_name, include_metadata=True)
        assert(result.shape[0] == sum([len(x.split()) for x in self.test_strings]))
        assert(result.shape[1] == 3)
        assert(result.index[0] == 0 and result.index[-1] == sum([len(x.split()) for x in self.test_strings]) - 1)

        assert((result[result['sent_id'] == 0].loc[:,metadata_col_name] == 1).all())
        assert((result[result['sent_id'] == 1].loc[:,metadata_col_name] == 0).all())
        assert((result[result['sent_id'] == 2].loc[:,metadata_col_name] == -1).all())

        # Test with list parameter.
        result_metadata_list = ngram_generation.generate_corpus_ngrams(test_df, test_col_name, include_metadata=[metadata_col_name])
        assert(result_metadata_list.shape[0] == sum([len(x.split()) for x in self.test_strings]))
        assert(result_metadata_list.shape[1] == 3)
        assert(result_metadata_list.index[0] == 0 and result_metadata_list.index[-1] == sum([len(x.split()) for x in self.test_strings]) - 1)

        assert((result_metadata_list[result_metadata_list['sent_id'] == 0].loc[:,metadata_col_name] == 1).all())
        assert((result_metadata_list[result_metadata_list['sent_id'] == 1].loc[:,metadata_col_name] == 0).all())
        assert((result_metadata_list[result_metadata_list['sent_id'] == 2].loc[:,metadata_col_name] == -1).all())

        assert(((result == result_metadata_list).all()).all())

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
