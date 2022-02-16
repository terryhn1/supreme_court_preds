import pandas as pd
import re
import torch
import numpy as np
import string
from sklearn.model_selection import train_test_split
from nltk import word_tokenize


class JudgmentDataset(torch.utils.data.Dataset):

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    def letterToIndex(letter):
        return JudgmentDataset.all_letters.find(letter)

    def letterToTensor(letter):
        tensor = torch.zeros(1, JudgmentDataset.n_letters)
        tensor[0][JudgmentDataset.letterToIndex(letter)] = 1
        return tensor

    def lineToTensor(line):
        tensor = torch.zeros(len(line), 1, JudgmentDataset.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][JudgmentDataset.letterToIndex(letter)] = 1
        return tensor

    def __init__(self, judg_dataframe):
        x = judg_dataframe["textdata"]
        y = judg_dataframe["winner_label"]

        self.y_train = torch.tensor(y, dtype = torch.float32)
        self.x_train = JudgmentDataset.lineToTensor(x)

        self.targets = judg_dataframe["winner_label"]
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return {'text': self.x_train[idx], 'label': self.y_train[idx], 'text_length': len(self.x_train[idx])}



def extract_data():
    """Takes the information from the CSV File and packages into a Pandas Dataframe for further usage"""    


    data = pd.read_csv('csv_data/justice.csv', encoding= "utf-8")

    cases = list()
    clean = re.compile('<.*?>')
    #Takes out data for future analysis
    for i in range(len(data)):
        textdata = re.sub(clean, '', data["facts"][i]).strip()

        cases.append({"textdata": textdata, "seq_len": data["facts_len"][i], "id": data["ID"][i], "case_name": data["name"][i], "first_party": data["first_party"][i],
        "second_party": data["second_party"][i], "winner_label": 1 if data["first_party_winner"][i] else 0,
        "timeline": data["term"][i], "decision_type": data["decision_type"][i]
        })

    #Converts the dictionary into a practical and usable dataset for Pytorch
    #the column 'first_party_winner' represents the label that should be predicted
    dataset = pd.DataFrame(cases)

    #converts the textdata into a corpus for word embedding analysis with each case representing its own space originally
    text_corpus = [word_tokenize(case["textdata"]) for case in cases]

    return dataset, text_corpus

def clean_text(text):
    """Takes out the stopwords and the punctuations from the data before feeding it into gloVe vector dictionary"""


    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves"]

    for word in stopwords:
        text = text.replace(word, "")

    for punct in [".", ",", "!", "?"]:
        text = text.replace(punct, "")

    
    return text


def truncate_data(seq_length, dataset):
    pass

if __name__ == "__main__":
    dataset, text_corpus = extract_data()

    
