import math
import unittest
from pipeline import Pipeline
import pandas as pd

from utilities.spacy_utilities import Spacy_Manager

class PipelineTests(unittest.TestCase):
    # Set up and helper functions
    def setUp(self) -> None:
        test_data = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5]
        ]
        test_secondary_data = [
            [10],
            [11],
            [12],
            [13],
            [14],
            [15]
        ]
        self.test_df = pd.DataFrame(test_data)
        self.secondary_test_df = pd.DataFrame(test_secondary_data, columns=['a'])
        return super().setUp()
    
    @staticmethod
    def simple_extraction_fn(df: pd.DataFrame) -> pd.DataFrame:
        return df

    # Test functions
    def test_standard_configuration(self):
        pre_extraction_fns = [
            lambda x: x + 1,
            lambda x: x * 2
        ]
        post_extraction_fns = [
            lambda x: x - 1
        ]
        batch_size = 3

        def save_fn(df: pd.DataFrame):
            assert(df.shape[1] == self.test_df.shape[1])
            assert(df.shape[0] == self.test_df.shape[0])
        
        p = Pipeline(
            data_save_fn=save_fn,
            pre_extraction_fns=pre_extraction_fns,
            feature_extraction_fn=PipelineTests.simple_extraction_fn,
            post_extraction_fns=post_extraction_fns,
            batch_size=batch_size
        )
        p.start([self.test_df.copy(deep=True)])

    def test_no_processing_fns(self):
        pre_extraction_fns = []
        post_extraction_fns = []
        batch_size = 3

        def save_fn(df: pd.DataFrame):
            assert(df.shape[1] == self.test_df.shape[1])
            assert(df.shape[0] == self.test_df.shape[0])
        
        p = Pipeline(
            data_save_fn=save_fn,
            pre_extraction_fns=pre_extraction_fns,
            feature_extraction_fn=PipelineTests.simple_extraction_fn,
            post_extraction_fns=post_extraction_fns,
            batch_size=batch_size
        )
        p.start([self.test_df.copy(deep=True)])

    def test_multiple_dataframes(self):
        pre_extraction_fns = [
            lambda x: x + 1,
            lambda x: x * 2
        ]
        post_extraction_fns = [
            lambda x: x - 1
        ]
        batch_size = 3

        def save_fn(df: pd.DataFrame):
            assert(df.shape[1] == self.test_df.shape[1] + self.secondary_test_df.shape[1])
            assert(df.shape[0] == self.test_df.shape[0])
            assert(df.shape[0] == self.secondary_test_df.shape[0])
        
        p = Pipeline(
            data_save_fn=save_fn,
            pre_extraction_fns=pre_extraction_fns,
            feature_extraction_fn=PipelineTests.simple_extraction_fn,
            post_extraction_fns=post_extraction_fns,
            batch_size=batch_size
        )
        p.start([self.test_df.copy(deep=True)], [iter([self.secondary_test_df.copy(deep=True)])])
            
    def test_split_df(self):
        batch_size = 4
        p = Pipeline(
            data_save_fn=None,
            pre_extraction_fns=[],
            feature_extraction_fn=PipelineTests.simple_extraction_fn,
            post_extraction_fns=[],
            text_column_name='',
            ngram_column_name='',
            batch_size=batch_size
        )

        res = p._split_df(self.test_df)
        res_concat = pd.concat(res, axis=0, ignore_index=True)
        assert(len(res) == math.ceil(self.test_df.shape[0] / batch_size))
        assert(sum([len(x) for x in res]) == self.test_df.shape[0])
        assert(res_concat.shape[0] == self.test_df.shape[0])
        assert(res_concat.shape[1] == self.test_df.shape[1])
        assert((res_concat == self.test_df).all(axis=None))
            
    def test_split_sp_docs(self):
        batch_size = 4
        test_sp_docs = list(Spacy_Manager.generate_docs([
            "Lorem ipsum dolor sit amet consectetur adipiscing",
            "elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
            "Ut enim ad minim veniam quis nostrud exercitation",
            "ullamco laboris nisi ut aliquip ex ea commodo consequat",
            "Duis aute irure dolor in reprehenderit in voluptate",
            "velit esse cillum dolore eu fugiat nulla pariatur"
        ]))

        p = Pipeline(
            data_save_fn=None,
            pre_extraction_fns=[],
            feature_extraction_fn=PipelineTests.simple_extraction_fn,
            post_extraction_fns=[],
            text_column_name='',
            ngram_column_name='',
            batch_size=batch_size
        )

        res = p._split_sp_docs(test_sp_docs)
        assert(len(res) == math.ceil(len(test_sp_docs) / batch_size))
        assert(sum([len(x) for x in res]) == len(test_sp_docs))

        