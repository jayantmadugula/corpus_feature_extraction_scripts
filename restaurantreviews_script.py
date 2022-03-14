'''
This script runs the Restaurant Reviews dataset
through the Pipeline with a pre-determined set of
pre-processing and post-processing functions.

Citation for the Restaurant Reviews data:
Beyond the stars: improving rating predictions using review text content.
G Ganu, N Elhadad, A Marian. Proc. WebDB. 1-6. 2009.
'''

import json
import sqlite3
from functools import partial
from utilities.database_utilities import load_df, save_df
from processing_functions import ngram_generation, text_preprocessing as tp
from pipeline import Pipeline


if __name__ == '__main__':
    # Load parameters.
    with open('./parameters.json') as params_fp:
        params = json.load(params_fp)
    
    n_processes = params['num_processes']
    batch_size = params['batch_size']

    database_path = params['restaurant_reviews']['database_path']
    table_name = params['restaurant_reviews']['text_table_name']

    window_len = 2 # len(ngram) = (2 * window_len) + 1
    output_table_name = f'restaurantreviews_n={window_len}'

    # Logging
    log_dict = dict()
    log_dict['Pipeline Input'] = {
        'Database Path': database_path,
        'Table Name': table_name
    }
    log_dict['ngram Size'] = f'{window_len}'

    log_dict['Pipeline Output'] = {
        'Table Name', output_table_name
    }
    run_name = output_table_name

    # Get data iterator.
    conn = sqlite3.connect(database_path)
    sql_iter = load_df(conn, table_name, chunksize=batch_size)
    print(type(sql_iter))

    # Call Pipeline with data and processing functions.
    ngram_extraction_fn = partial(ngram_generation.generate_corpus_ngrams, col_name='review_spdocs', n=window_len)
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
        text_column_name='review',
        ngram_column_name='ngram',
        batch_size=batch_size,
        num_processes=n_processes,
        use_spacy=True,
        log_dict=log_dict,
        run_name=run_name
    )
    p.start(sql_iter)

    conn.close()