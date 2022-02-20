from typing import Callable, Iterable
import pandas as pd
import sqlite3

class Pipeline():
    '''
    This class defines a Pipeline object that uses generators
    and batch processing to efficiently apply a series of 
    normalization and feature extraction operations to a 
    text-based dataset.
    
    Pipeline objects apply operations on pandas DataFrames. 
    '''

    def __init__(
        self,
        data_generator: Iterable[pd.DataFrame],
        data_save_fn: Callable[[pd.DataFrame], None],
        pre_extraction_fns: Iterable[Callable[[pd.DataFrame], pd.DataFrame]],
        feature_extraction_fn: Callable[[pd.DataFrame], pd.DataFrame],
        post_extraction_fns: Iterable[Callable[[pd.DataFrame], pd.DataFrame]]
        ):
        self._data_generator = data_generator
        self._data_save_fn = data_save_fn
        self._pre_extraction_fns = pre_extraction_fns
        self._feature_extraction_fn = feature_extraction_fn
        self._post_extraction_fns = post_extraction_fns

    def _process(self, df: pd.DataFrame) -> None:
        print(f'Processing DataFrame with shape: {df.shape}')

    def start(self):
        for (i, current_df) in enumerate(self._data_generator):
            # TODO: Multi-processing happens here, _process() handles a single DataFrame
            self._process(current_df)
            print(f'Pipeline step {i} complete.')
        print('Pipeline complete.')


if __name__ == '__main__':
    sqlite_conn = sqlite3.connect('../databases/corpus_database.db')
    x = pd.read_sql(
        sql='SELECT * FROM semeval16_reviews', 
        con=sqlite_conn,
        index_col='index',
        chunksize=100)
    print(type(x))

    p = Pipeline(x, None, [], None, [])
    p.start()