from distutils.command.build import build
import json
from cv2 import mean
import pandas as pd
import re
import torch
import numpy as np
import string
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
def extract_data():
    """Takes the information from the CSV File and packages into a Pandas Dataframe for further usage"""    


    data = pd.read_csv('csv_data/justice.csv', encoding= "utf-8")

    cases = list()
    clean = re.compile('<.*?>')
    #Takes out data for future analysis
    for i in range(len(data)):
        textdata = re.sub(clean, '', data["facts"][i]).strip()

        cases.append({"textdata": textdata, "case_name":data["name"][i],"first_party": data["first_party"][i], "second_party": data["second_party"][i], "label": 1 if data["first_party_winner"][i] else 0})

    #Converts the dictionary into a practical and usable dataset for Pytorch
    #the column 'first_party_winner' represents the label that should be predicted
    dataset = pd.DataFrame(cases)

    #converts the textdata into a corpus for word embedding analysis with each case representing its own space originally
    text_corpus = [word_tokenize(case["textdata"]) for case in cases]

    return dataset, text_corpus

def extract_data_EU():
    with open('jsonl/dev.jsonl') as f:
        euDevCourtData = [json.loads(line) for line in f]
    with open('jsonl/train.jsonl') as f1:
        euTrainCourtData = [json.loads(line) for line in f1]
    with open('jsonl/test.jsonl') as f2:
        euTestCourtData = [json.loads(line) for line in f2]


    cases = list()
    clean = re.compile('<.*?>')

    for each in euDevCourtData:
        string_textdata = ""
        for fact in each['facts']:
            string_textdata += fact[3:]
        textdata = re.sub(clean, '', string_textdata).strip()
        favored = ''
        if len(each["violated_articles"]) > 0:
            favored = 0
        else:
            favored = 1
        cases.append({"textdata":textdata, "case_name": each['title'], "first_party": ",".join(each['applicants']), "second_party": ",".join(each["defendants"]), "label": favored})

    for each in euTrainCourtData:
        string_textdata = ""
        for fact in each['facts']:
            string_textdata += fact[3:]
        textdata = re.sub(clean, '', string_textdata).strip()
        favored = ''
        if len(each["violated_articles"]) > 0:
            favored = 0
        else:
            favored = 1
        cases.append({"textdata": textdata, "case_name": each['title'], "first_party": ",".join(each['applicants']), "second_party": ",".join(each["defendants"]), "label": favored})
    
    for each in euTestCourtData:
        string_textdata = ""
        for fact in each['facts']:
            string_textdata += fact[3:]
        textdata = re.sub(clean, '', string_textdata).strip()
        favored = ''
        if len(each["violated_articles"]) > 0:
            favored = 0
        else:
            favored = 1
        cases.append({"textdata": textdata, "case_name": each['title'], "first_party": ",".join(each['applicants']), "second_party": ",".join(each["defendants"]), "label": favored})

    #Truncating the textdata: range of mean to max
    length_list = [len(case["textdata"]) for case in cases]
    mean_length = int(sum(length_list) / len(cases))
    new_cases = remove_outliers(cases)
    print(len(new_cases))
    truncate_text(cases, mean_length)
    return new_cases


def remove_outliers(data, min_length = 1000, max_length = 30000):
    #returns a new list to remove outlier cases that might not be optimal for computation for training or testing
    copy = list()
    for case in data:
        if len(case["textdata"]) < max_length and len(case["textdata"]) > min_length:
            copy.append(case)
    
    return copy

def truncate_text(data, max_length):
    #modifier to truncate the text data inside
    for i in range(len(data)):
        if len(data[i]["textdata"]) > max_length:
            data[i]["textdata"] = data[i]["textdata"][:max_length]
    

if __name__ == "__main__":
    cases = extract_data_EU()
    dataframe = pd.DataFrame(cases)
    dataframe.to_csv('csv_data/eu_human_rights.csv')

    
