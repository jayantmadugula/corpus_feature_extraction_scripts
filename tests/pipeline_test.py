import unittest
from pipeline import Pipeline

class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        # TODO: Create a DataFrame
        # TODO: Define a series of basic functions for testing Pipeline object
        return super().setUp()

    def test_basic_pipeline(self):
        p = Pipeline(None, lambda f: None, [], lambda j: None, [])