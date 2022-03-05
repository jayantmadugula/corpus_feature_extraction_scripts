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
from utilities.database_utilities import load_df
from pipeline import Pipeline


if __name__ == '__main__':
    with open('./parameters.json') as params_fp:
        params = json.load(params_fp)
    
    n_processes = params['num_processes']
    batch_size = params['batch_size']

    database_path = params['restaurant_reviews']['database_path']
    table_name = params['restaurant_reviews']['text_table_name']

    conn = sqlite3.connect(database_path)
    sql_iter = load_df(conn, table_name, chunksize=batch_size)
    print(type(sql_iter))
    conn.close()