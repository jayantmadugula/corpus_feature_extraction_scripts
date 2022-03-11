'''
This file contains a few simple functions useful for loading
tables from a SQLite database.
'''

import sqlite3
import pandas as pd

def load_table(conn: sqlite3.Connection, table_name: str) -> sqlite3.Cursor:
    cur = conn.cursor()
    return cur.execute("SELECT * FROM {}".format(table_name))

def load_df(conn: sqlite3.Connection, table_name: str, index_col = 'index', chunksize: int = 1) -> sqlite3.Cursor:
    '''
    Returns an iterator of DataFrames.
    '''
    return pd.read_sql(
        'SELECT * FROM {}'.format(table_name), 
        conn, 
        index_col = index_col,
        chunksize = chunksize)

def save_df(df: pd.DataFrame, conn: sqlite3.Connection, table_name: str):
    '''
    Saves incoming `pd.DataFrame` to a SQLite3 database.
    If the table already exists, it will be appended to.
    '''
    df.to_sql(table_name, conn, if_exists='append')