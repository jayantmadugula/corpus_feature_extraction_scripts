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
        self.test_df = pd.DataFrame(test_data)
        return super().setUp()
    
    @staticmethod
    def extraction_fn(df: pd.DataFrame) -> pd.DataFrame:
        return df

    # Test functions
    def test_basic_pipeline(self):
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
            feature_extraction_fn=PipelineTests.extraction_fn,
            post_extraction_fns=post_extraction_fns,
            batch_size=batch_size
        )
        p.start([self.test_df.copy(deep=True)])