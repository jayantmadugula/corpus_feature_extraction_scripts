# NLP Corpus Feature Extraction

This repository contains two main sets of work: a `Pipeline` class that provides easy-to-use batch processing and parallelization and a set of ngram generation scripts that use `Pipeline` for four different datasets.

## Pipeline
First, `Pipeline.py` contains a class, `Pipeline` that can be used to efficiently process large amounts of text data using processing and featurization functions that are passed in. It uses both batch processing and multiprocessing to improve performance and avoid "out of memory" errors that might occur otherwise. As each batch is run through the Pipeline, it is saved and removed from memory before the next batch of data is read and processed. `Pipeline` objects support three types of functions:

1. Pre-processing functions: these are applied to the input text first. 
2. Feature extraction function: this function is intended to be used for ngram generation (or similar).
3. Post-processing functions: these are applied to the result of the feature extraction function.

Batch processing and multiprocessing are handled automatically. Simply pass in functions that perform the requested work when creating a `Pipeline` instance and use the `.start()` function to begin processing the input data. A "save" function, which defines where the output from the `Pipeline` should go, must also be passed into the `Pipeline`. Finally, `Pipeline` will save a log with some basic information about each run to a single JSON log file.

## Scripts
Secondly, there are four Python script files in this repository. Each script is targeted at a different NLP dataset and assumes the raw text data is saved to a SQLite3 database. A few required parameters, including database path and table name, for each script can be found in `parameters.json`. Each of these scripts uses `Pipeline` for the text processing and ngram generation. For more information about these datasets and how the raw data was originally saved to a SQLite3 database, please look at these [corpus parsing scripts](https://github.com/jayantmadugula/corpus_parsing_scripts) I wrote.

## Other Work
The remaining work in this repository is the functions defined specifically for the four scripts, including an ngram generation function that is able to save correlated metadata alongside a newly generated ngram. `spaCy` is also used to help with part-of-speech tagging, allowing the ngram generation function to only create ngrams when the central word in the ngram has a specified tag. 

## Required Packages
This is (currently) not guaranteed to be an exhaustive list (sorry!). However, these are the main packages you'll need to install to run the included scripts and use `Pipeline`.

Packages:

- pandas
- numpy
- [spaCy](https://spacy.io/usage)
- [NLTK](https://www.nltk.org/install.html)

**Note**: After installing spaCy, please run `python -m spacy download en_core_web_lg`.