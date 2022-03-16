import json
from multiprocessing import Pool
import datetime
from typing import Callable, Iterable, Iterator, List
import pandas as pd
from utilities.spacy_utilities import Spacy_Manager
from utilities.logging_utilities import get_fn_name

DEFAULT_OUTPUT_LOG_PATH = './pipeline_log.json'
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
        text_column_name: str,
        ngram_column_name: str,
        **kwargs
        ):
        self._data_save_fn = data_save_fn
        self._pre_extraction_fns = pre_extraction_fns
        self._feature_extraction_fn = feature_extraction_fn
        self._post_extraction_fns = post_extraction_fns
        self._input_column_name = text_column_name
        self._feature_column_name = ngram_column_name

        # Process additional keyword arguments.
        self._batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
        self._num_processes = kwargs['num_processes'] if 'num_processes' in kwargs else None
        self._use_spacy = kwargs['use_spacy'] if 'use_spacy' in kwargs else False

        # Logging
        self._log_path: str = kwargs['log_filepath'] if 'log_filepath' in kwargs else DEFAULT_OUTPUT_LOG_PATH
        self._pipeline_log: dict[str, str] = kwargs['log_dict'] if 'log_dict' in kwargs else {'Pipeline Input': 'None'}
        self._run_name: str = kwargs['run_name'] if 'run_name' in kwargs else str(datetime.datetime.now())
        self._create_log()

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f'Processing DataFrame with shape: {df.shape}')
        # Run pre-extraction functions.
        for fn in self._pre_extraction_fns:
            try:
                df.loc[:, self._input_column_name] = fn(df.loc[:, self._input_column_name])
            except BaseException:
                print(f'Pre-extraction function {fn.__name__} failed with an unexpected error.')
                raise
        
        # Run feature extraction function using multiprocessing.
        if self._use_spacy:
            # TODO Investigate taking advantage of spaCy's Doc generator.
            docs_col_name = '{}_spdocs'.format(self._input_column_name)
            df.loc[:, docs_col_name] = list(Spacy_Manager.generate_docs(df.loc[:, self._input_column_name]))
        batched_dfs = self._split_df(df)
        
        pool_size = self._num_processes if self._num_processes is not None else 1
        with Pool(pool_size) as p:
            try:
                res = p.map(self._feature_extraction_fn, batched_dfs)
            except BaseException:
                print(f'Feature extraction function {self._feature_extraction_fn.__name__} failed with an unexpected error.')
                raise
        feature_df = pd.concat(res, ignore_index=True, axis=0)

        # Run post-extraction functions.
        for fn in self._post_extraction_fns:
            try:
                feature_df.loc[:, self._feature_column_name] = fn(feature_df.loc[:, self._feature_column_name])
            except BaseException:
                print(f'Post-extraction function {fn.__name__} failed with an unexpected error.')
                raise

        return feature_df

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
        self._save_log()

    # Logging
    def _create_log(self):
        self._pipeline_log['Pre-Extraction Functions'] = [get_fn_name(f) for f in self._pre_extraction_fns]
        self._pipeline_log['Feature Extraction Function'] = get_fn_name(self._feature_extraction_fn)
        self._pipeline_log['Post-Extraction Functions'] = [get_fn_name(f) for f in self._post_extraction_fns]

        self._pipeline_log['Pipeline Settings'] = {
            'Using spaCy': f'{self._use_spacy}',
            'Batch Size': f'{self._batch_size}',
        }

        self._pipeline_log['Input Text Column Name'] = self._input_column_name
        self._pipeline_log['Output Column Name'] = self._feature_column_name

        self._pipeline_log['Start Time'] = str(datetime.datetime.now())

    def _save_log(self):
        self._pipeline_log['End Time'] = str(datetime.datetime.now())
        with open(self._log_path, mode='r') as fp:
            try:
                existing_log = json.load(fp)
            except:
                existing_log = dict()
        
            if existing_log is None:
                existing_log[self._run_name] = self._pipeline_log
                json.dump(
                    existing_log, 
                    fp)
            else:
                existing_log[self._run_name] = self._pipeline_log
        
        with open(self._log_path, mode='w') as fp_w:
            json.dump(existing_log, fp_w)
