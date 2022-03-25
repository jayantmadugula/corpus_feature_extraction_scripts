'''
Functions to help validate inputs.
'''

from typing import Iterable


def validate_spacy_pos(pos_list: Iterable[str]):
    valid_pos = set(['ADV', 'NOUN', 'PRON', 'PROPN', 'VERB', 'ADJ'])

    invalid_pos = set()
    for pos in pos_list:
        if pos not in valid_pos:
            invalid_pos.add(pos)
    
    if len(invalid_pos) > 0:
        raise ValueError(f'Invalid part-of-speech provided: {invalid_pos}')