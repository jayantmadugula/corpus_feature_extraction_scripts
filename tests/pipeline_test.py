import unittest
from pipeline import Pipeline
import pandas as pd

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
            