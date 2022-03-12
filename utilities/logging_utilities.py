'''
Contains functions to help with logging.
'''

def get_fn_name(fn) -> str:
    try:
        return fn.__name__
    except:
        return 'None'