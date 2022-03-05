from multiprocessing import Pool
from typing import Callable, Iterable, Iterator, List
import pandas as pd

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
        data_save_fn: Callable[[pd.DataFrame], None],
        pre_extraction_fns: Iterable[Callable[[pd.DataFrame], pd.DataFrame]],
        feature_extraction_fn: Callable[[pd.DataFrame], pd.DataFrame],
        post_extraction_fns: Iterable[Callable[[pd.DataFrame], pd.DataFrame]],
        **kwargs
        ):
        self._data_save_fn = data_save_fn
        self._pre_extraction_fns = pre_extraction_fns
        self._feature_extraction_fn = feature_extraction_fn
        self._post_extraction_fns = post_extraction_fns

        # Process additional keyword arguments.
        self._batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
        self._num_processes = kwargs['num_processes'] if 'num_processes' in kwargs else None

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f'Processing DataFrame with shape: {df.shape}')
        # Run pre-extraction functions.
        for fn in self._pre_extraction_fns:
            try:
                df = fn(df)
            except BaseException:
                print(f'Pre-extraction function {fn.__name__} failed with an unexpected error.')
                raise
        
        # Run feature extraction function using multiprocessing.
        batched_dfs = self._split_df(df)
        pool_size = self._num_processes if self._num_processes is not None else 1
        with Pool(pool_size) as p:
            try:
                res = p.map(self._feature_extraction_fn, batched_dfs)
            except BaseException:
                print(f'Feature extraction function {self._feature_extraction_fn.__name__} failed with an unexpected error.')
                raise
        df = pd.concat(res, ignore_index=True, axis=0)

        # Run post-extraction functions.
        for fn in self._post_extraction_fns:
            try:
                df = fn(df)
            except BaseException:
                print(f'Post-extraction function {fn.__name__} failed with an unexpected error.')
                raise

        return df

    def _split_df(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        '''
        Splits a DataFrame into batches based on the Pipeline's batch size.
        '''
        if self._batch_size is None or self._batch_size >= df.shape[0]: return [df]

        batched_dfs = []
        for i in range(self._batch_size, df.shape[0], self._batch_size):
            batched_dfs.append(df.iloc[i - self._batch_size:i])
        batched_dfs.append(df.iloc[i:])
        
        return batched_dfs

    def start(
        self, 
        df_generator: Iterable[pd.DataFrame], 
        additional_df_generators: Iterable[Iterator[pd.DataFrame]] = []):
        for (i, current_df) in enumerate(df_generator):
            if len(additional_df_generators) > 0:
                # Join all DataFrames by index
                current_additional_dfs = list(map(next, additional_df_generators))
                current_df = current_df.join(
                    current_additional_dfs,
                    how='inner')

            processed_df = self._process(current_df)
            self._data_save_fn(processed_df)

            print(f'Pipeline step {i} complete.')
        
        print('Pipeline complete.')