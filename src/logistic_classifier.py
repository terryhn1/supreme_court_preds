import pandas as pd
import re
from nltk import word_tokenize
import sklearn
import json
from sklearn.feature_extraction.text import * 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model 
from sklearn import metrics 

def text_extraction():
    data = pd.read_csv('datasets/csv_data/justice.csv')
    with open('./datasets/jsonl/dev.jsonl') as f:
        euDevCourtData = [json.loads(line) for line in f]
    with open('./datasets/jsonl/train.jsonl') as f1:
        euTrainCourtData = [json.loads(line) for line in f1]
    with open('./datasets/jsonl/test.jsonl') as f2:
        euTestCourtData = [json.loads(line) for line in f2]


    cases = list()
    clean = re.compile('<.*?>')


    #Takes out data for future analysis
    '''
    for i in range(len(data)):
        textdata = re.sub(clean, '', data["facts"][i]).strip()

        cases.append({"textdata": textdata, "id": data["ID"][i], "case_name": data["name"][i], "first_party": data["first_party"][i],
        "second_party": data["second_party"][i], "favored": data["first_party"][i] if data["first_party_winner"][i] else data["second_party"][i],
        "timeline": data["term"][i], "decision_type": data["decision_type"][i]
        })'''

    for each in euDevCourtData:
        string_textdata = ""
        for fact in each['facts']:
            string_textdata += fact[3:]
        textdata = re.sub(clean, '', string_textdata).strip()
        favored = ''
        if len(each["violated_articles"]) > 0:
            favored = each['applicants']
        else:
            favored = each['defendants']
        cases.append({"textdata": textdata, "case_name": each['title'], "first_party": each['applicants'], "second_party": each["defendants"], "favored": favored, "timeline": each["judgment_date"].split('-')[0]})

    for each in euTrainCourtData:
        string_textdata = ""
        for fact in each['facts']:
            string_textdata += fact[3:]
        textdata = re.sub(clean, '', string_textdata).strip()
        favored = ''
        if len(each["violated_articles"]) > 0:
            favored = each['applicants']
        else:
            favored = each['defendants']
        cases.append({"textdata": textdata, "case_name": each['title'], "first_party": each['applicants'], "second_party": each["defendants"], "favored": favored, "timeline": each["judgment_date"].split('-')[0]})
    
    for each in euTestCourtData:
        string_textdata = ""
        for fact in each['facts']:
            string_textdata += fact[3:]
        textdata = re.sub(clean, '', string_textdata).strip()
        favored = ''
        if len(each["violated_articles"]) > 0:
            favored = each['applicants']
        else:
            favored = each['defendants']
        cases.append({"textdata": textdata, "case_name": each['title'], "first_party": each['applicants'], "second_party": each["defendants"], "favored": favored, "timeline": each["judgment_date"].split('-')[0]})
    '''print(data["first_party_winner"])
    df_target = data["first_party_winner"]
    df_target.reset_index(drop=True, inplace=True)
    print(df_target)'''
    return cases


# Takes a list of terms to extract and add to the Bag of Words extraction, list of dictionaries which has case details, time frame start date, and time frame end date
def BOW(extraction_terms, case_details, start_timeline = 1776, end_timeline = 2022):
    text = []
    document = ''
    Y = []
    for case in case_details:
        if '-' in case['timeline']:
            resolution_date = int(case['timeline'].split('-')[1])
        else:
            resolution_date = int(case['timeline'])
        if resolution_date > start_timeline and resolution_date < end_timeline:
            document = ''
            for term in extraction_terms:
                document += str(case[term]) + ' '
            if (case['first_party'] == case['favored']):
                Y.append(1)
            else:
                Y.append(0)
            text.append(document)
        
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=.01)
    X = vectorizer.fit_transform(text)


    return X, Y

def logistic_classification(X, Y, test_fraction):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)

    print('Number of training examples: ', X_train.shape[0])
    print('Number of testing examples: ', X_test.shape[0])   
    print('Vocabulary size: ', X_train.shape[1]) 

    #classifier = linear_model.LogisticRegression( penalty='l1', solver='liblinear')  
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    # Train a logistic regression classifier and evaluate accuracy on the training data
    print('\nTraining a model with', X_train.shape[0], 'examples.....')
    classifier.fit(X_train, Y_train) 
    train_predictions = classifier.predict(X_train)	 # Training
    train_accuracy = metrics.accuracy_score(Y_train, train_predictions)
    class_probabilities_train = classifier.predict_proba(X_train)
    train_auc_score = metrics.roc_auc_score(Y_train, class_probabilities_train[:, 1])
    print('\nTraining:')
    print(' accuracy:',format( 100*train_accuracy , '.2f') ) 
    print(' AUC value:', format( 100*train_auc_score , '.2f') )

    # Compute and print accuracy and AUC on the test data
    print('\nTesting: ')
    test_predictions = classifier.predict(X_test)	 
    test_accuracy = metrics.accuracy_score(Y_test, test_predictions) 
    print(' accuracy:', format( 100*test_accuracy , '.2f') )

    class_probabilities = classifier.predict_proba(X_test)
    test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities[:, 1])
    print(' AUC value:', format( 100*test_auc_score , '.2f') )

    return classifier

def predict(Text, classifier):
    pass


if __name__ == "__main__":
    extraction_terms = ['textdata', 'first_party', 'second_party']
    cases = text_extraction()
    X, Y = BOW(extraction_terms, cases)
    print(X.shape)
    print(len(Y))

    classifier = logistic_classification(X, Y, .3)