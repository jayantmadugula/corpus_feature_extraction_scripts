'''
This script runs the SOCC dataset
through the Pipeline with a pre-determined set of
pre-processing and post-processing functions.

Citation for SOCC dataset: 
Kolhatkar, Varada, et al. "The SFU opinion and comments corpus: A corpus for the analysis of online news comments." Corpus Pragmatics 4.2 (2020): 155-190.
APA
'''

import argparse
from functools import partial
import json
import sqlite3
from pipeline import Pipeline
from processing_functions import ngram_generation, text_preprocessing as tp
from utilities.database_utilities import load_df, remove_existing_table, save_df

from utilities.input_validation_utilities import validate_spacy_pos


if __name__ == '__main__':
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='''
        Normalizes, processes, and extracts ngrams from the SOCC dataset.
        
        The raw text data is expected to be saved to a SQLite3 database. Settings for this
        scripts can be found in the "socc" section of `parameters.json`.
        ''')
    parser.add_argument(
        'ngram_context_size', 
        metavar='N', 
        type=int, 
        default=2,
        nargs='?',
        help='the size of the resulting ngrams will be 2 * N + 1')
    parser.add_argument(
        'use_pos_filtering',
        metavar='P',
        type=bool,
        default=False,
        nargs='?',
        help='''
            if set to True, ngram generation will only occur for ngrams centered on words
            that are defined in parameters.json
            '''
    )

    args = parser.parse_args()
    window_len =  args.ngram_context_size # len(ngram) = (2 * window_len) + 1
    use_pos_filtering = args.use_pos_filtering

    # Load parameters.
    with open('./parameters.json') as params_fp:
        params = json.load(params_fp)

    n_processes = params['num_processes']
    batch_size = params['batch_size']

    database_path = params['socc']['database_path']
    table_name = params['socc']['text_table_name']
    text_column_name = params['socc']['text_column_name']

    pos_filter = params['socc']['pos_filter_list']
    validate_spacy_pos(pos_filter)

    output_table_name = f'socc={window_len}'
    if use_pos_filtering: output_table_name = f'{output_table_name}_pos-filter'

    # Remove pre-existing table if necessary.
    conn = sqlite3.connect(database_path)
    remove_existing_table(output_table_name, conn)

    # Logging
    log_dict = dict()
    log_dict['Pipeline Input'] = {
        'Database Path': database_path,
        'Table Name': table_name,
        'Include PoS Filtering': use_pos_filtering
    }
    log_dict['ngram Size'] = f'{window_len}'

    log_dict['Pipeline Output'] = {
        'Table Name': output_table_name
    }

    run_name = output_table_name

    # Get data iterator.
    sql_iter = load_df(conn, table_name, chunksize=batch_size)

    # Call Pipeline with data and processing functions.
    included_metadata_columns = [
        'article_id'
    ]
    if use_pos_filtering:
        ngram_extraction_fn = partial(
            ngram_generation.generate_corpus_ngrams, 
            col_name=f'{text_column_name}_spdocs', 
            n=window_len,
            include_metadataa=included_metadata_columns,
            pos_filter=pos_filter)
    else:
        ngram_extraction_fn = partial(
            ngram_generation.generate_corpus_ngrams, 
            col_name=f'{text_column_name}_spdocs', 
            n=window_len,
            include_metadata=included_metadata_columns)
    ngram_extraction_fn.__name__ = ngram_generation.generate_corpus_ngrams.__name__

    partial_save_fn = partial(save_df, conn=conn, table_name=output_table_name)
    partial_save_fn.__name__ = save_df.__name__

    p = Pipeline(
        data_save_fn=partial_save_fn,
        pre_extraction_fns=[
            tp.remove_punctuation,
            tp.lowercase_words,
            tp.normalize_spacing
        ],
        feature_extraction_fn=ngram_extraction_fn,
        post_extraction_fns=[],
        text_column_name=text_column_name,
        ngram_column_name='ngram',
        batch_size=batch_size,
        num_processes=n_processes,
        use_spacy=True,
        log_dict=log_dict,
        run_name=run_name
    )
    p.start(sql_iter)

    conn.close()