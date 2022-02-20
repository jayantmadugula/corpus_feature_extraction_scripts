import unittest
from pipeline import Pipeline

class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_basic_pipeline(self):
        p = Pipeline(None, lambda f: None, [], lambda j: None, [])