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

def remove_existing_table(table_name: str, conn: sqlite3.Connection):
    '''
    Drops a table if it exists in the given SQLite3 database.
    '''
    sql_str = 'DROP TABLE "{}";'.format(table_name)
    cur = conn.cursor()
    try:
        cur.execute(sql_str)
        conn.commit()
    except sqlite3.OperationalError as e:
        err_message: str = e.args[0]
        if 'no such table' in err_message:
            print('The table does not already exist. Continuing without raising an exception.')
        else:
            raise e