#creating histogram for document lengths
#added fact_len to dataframe in extraction.py's extract_data() but changed it back so no issues arise later

from pydoc import doc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import extraction


def us_histogram():
    dataset, text_corpus = extraction.extract_data()

    fact_lengths = dataset['facts_len']

    doc_lens = []
    for i in fact_lengths:
        doc_lens.append(i)

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(doc_lens, bins = 10)

    ax.set_title("Supreme Court Document Lengths (Facts)")
    ax.set_xlabel('Document Length')
    ax.set_ylabel('Frequency')

    plt.show()


def eu_histogram_truncated():
    dataset = extraction.extract_data_EU()
    print(len(dataset[0]['textdata']))

    doc_lengths = []
    for i in range(len(dataset)):
        doc_lengths.append(len(dataset[i]['textdata']))

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(doc_lengths, bins = 8)

    ax.set_title("EU Court Document Lengths (Truncated Facts)")
    ax.set_xlabel('Document Length')
    ax.set_ylabel('Frequency')

    plt.show()

def eu_histogram_nontruncated():
    dataset = extraction.extract_data_EU()\

    doc_lengths = []
    for i in range(len(dataset)):
        doc_lengths.append(len(dataset[i]['textdata']))

    #print(doc_lengths)

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(doc_lengths, bins = 10)

    ax.set_title("EU Court Document Lengths (Nontruncated Facts)")
    ax.set_xlabel('Document Length')
    ax.set_ylabel('Frequency')

    plt.show()

if __name__ == "__main__":
    us_histogram()
    # eu_histogram_truncated()
    # eu_histogram_nontruncated()
