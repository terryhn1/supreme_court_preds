Libraries Used
- Pytorch(https://pytorch.org/)
- numpy(https://numpy.org/)
- Torchtext(https://pytorch.org/text/stable/index.html)
- TDQM(https://tqdm.github.io/docs/notebook/)
- gloVe(https://nlp.stanford.edu/projects/glove/)
- pandas(https://pandas.pydata.org/)
- scikit-learn(https://scikit-learn.org/stable/)

--------------------------------
Publicly Available Code
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
- Modified approximately 4 lines
- Added approximately 2 lines
- Removed approximately 32 lines
- Variations of training/testing code through different files: lstm.py, eu_train.py, us_train_and_eu_test.py, eu_train_and_combined_test.py
Assignment 1 Logistic Classifier(https://www.ics.uci.edu/~smyth/courses/cs175/assignment1_2022.html)
- listed as logistic_classifier.py
- modifies create_bow_from_review function and logistic_classification function
- Approximately 5 lines removed,8 lines modified, 4 lines added from create_bow_from_reviews function

--------------------------------
Original Code
- extraction.py(Reads CSV/JSON Files, truncates text data, removes outliers, creates labels for the EU Court of Human Rights dataset, creates pandas Dataframe from extracted data and converst to a new CSV file)
- histogram.py(Reads EU and Supreme Court files to create histograms for the number of documents and text data length)
