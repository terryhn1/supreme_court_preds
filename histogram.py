#creating histogram for document lengths
#added fact_len to dataframe in extraction.py's extract_data() but changed it back so no issues arise later

from pydoc import doc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import extraction

#dataset, text_corpus = extraction.extract_data()
dataset, text_corpus = extraction.extract_data_EU()

fact_lengths = dataset['facts_len']

doc_lens = []
for i in fact_lengths:
    doc_lens.append(i)

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(doc_lens, bins = 10)

ax.set_title("EU Court Document Lengths (Facts)")
ax.set_xlabel('Document Length')
ax.set_ylabel('Frequency')

plt.show()

