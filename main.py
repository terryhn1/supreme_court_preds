from tkinter import HIDDEN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchtext import datasets
import torch.optim as optim
import numpy as np
import time
import random
import torch
import torch.nn as nn
import sklearn
import extraction


#CONSTANTS for file reads
GLOVE50D = 'glove/glove.6B.50d.txt'
GLOVE100D = 'glove/glove.6B.100d.txt'
GLOVE200D = 'glove/glove.6B.200d.txt'
GLOVE300D = 'glove/glove.6B.200d.txt'

#CONSTANT FOR HIDDEN DIM/SIZE
HIDDEN_SIZE = 32

#BATCH SIZE FOR FORWARD PASSING
BATCH_SIZE = 32

#HOW MANY LSTM MODELS SHOULD BE STACKED
NUM_LAYERS = 2

#OUTPUT SIZE FOR LINEAR MODEL
OUTPUT_SIZE = 1

#TRAINING SIZE
TRAIN_SIZE = 0.8

#TESTING SIZE
TEST_SIZE = 0.2


class CaseSentimentLSTM(nn.Module):
    
    def __init__(self, weights_matrix, hidden_size, output_size, num_layers, drop_rate = 0.5):
        super(CaseSentimentLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.word_embeddings, num_embeddings , embedding_dim = create_emb_layer(weights_matrix)

        self.word_embeddings.weight.data.copy_(torch.tensor(weights_matrix))


        #2 Layered LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first = True, dropout = drop_rate)

        #Dropout Layer to prevent overfit
        self.dropout = nn.Dropout()

        #Fully Connected Linear Layer
        self.fc = nn.Linear(hidden_size, output_size)

        #Final Layer for Fact Positivity given the Party Label
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):

        #Embedding Layer
        out = self.word_embeddings(x)

        #LSTM Layer
        lstm_out, hidden  = self.lstm(out, hidden)

        #Dropout Layer
        out = self.dropout(lstm_out)
        
        #Fully Connected Layer
        out = self.fc(out)
        
        # Sigmoid Layer
        sig_out = self.sig(out)

        #reshape Sigmoid output to fit batch size
        sig_out = sig_out.view(BATCH_SIZE, -1)
        sig_out = sig_out[:,-1]


        # Return the propagated tensor
        return out, hidden

    def init_hidden_state(self, batch_size):
        #creates the h0 layer
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

def instantiate_model(weights_matrix, vocab_size):
    vocab_size = vocab_size + 1  # 0 Padding
    output_size = OUTPUT_SIZE
    hidden_dim = HIDDEN_SIZE
    n_layers = NUM_LAYERS

    net = CaseSentimentLSTM(weights_matrix,hidden_dim, output_size, n_layers)

    return net

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.mean()

    return acc

def train(model: CaseSentimentLSTM , iterator: torch.utils.data.DataLoader, optimizer: optim.Adam, criterion: nn.BCEWithLogitsLoss):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        
        text = batch["text"]
        text_length = batch["text_length"]

        preds = model(text, text_length).squeeze(1)

        loss = criterion(preds, batch["label"])

        acc = accuracy_score(preds, batch["label"])

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc /len(iterator)

def evaluate(model: CaseSentimentLSTM, iterator: torch.utils.data.DataLoader, criterion: nn.BCEWithLogitsLoss):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch["text"]
            text_length = batch["text_length"]

            preds = model(text, text_length).squeeze(1)

            loss = criterion(preds, batch["label"])

            acc = binary_accuracy(preds, batch["label"])

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss/len(iterator), epoch_acc/len(iterator)


#Initialize a dictionary to hold the values in the glove file
def initialize_glove():
    global glove
    glove = {}

    #Create a Timer to see the load time of the file
    start = time.time()

    with open(GLOVE100D, encoding= "utf-8") as file:
        for line in file:
            dimensions = line.split()
            word = dimensions[0]
            dimensions = np.asarray(dimensions[1:], dtype = "float32")
            glove[word] = dimensions

    end = time.time()
    print("Loading time for GLOVE:", end-start, " seconds")


def check_vocab_instances():
    vocab = set()
    emb_dim = len(glove["the"])
    dataset, corpus = extraction.extract_data()

    for case in corpus:
        for word in case:
            vocab.add(word)

    matrix_size = len(vocab)
    weights_matrix = np.zeros((matrix_size, emb_dim))
    words_found = 0

    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = glove[word]
            words_found +=1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale =0.6, size = (emb_dim,))
    
    return dataset, weights_matrix, words_found

def create_emb_layer(weights_matrix, non_trainable = False):
    print(type(weights_matrix))
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    
    if non_trainable:
        emb_layer.weight.requires_grad = False
    
    return emb_layer, num_embeddings, embedding_dim

if __name__ == "__main__":

    #SETTING THE SEED TO 1 SO NO RANDOM COMPUTATION
    torch.manual_seed(0)
    random.seed(0)

    #SETTING DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #INITIALIZING GLOVE
    initialize_glove()
    dataset, weights_matrix, words_found = check_vocab_instances()

    #CREATING DATASETS
    judg_data = extraction.JudgmentDataset(dataset)

    train_indices, test_indices, _, _ = train_test_split(range(len(judg_data)), judg_data.targets
                                                        , stratify = judg_data.targets, test_size= TEST_SIZE, random_state = 1)
    

    train_split = torch.utils.data.Subset(judg_data, train_indices)
    test_split = torch.utils.data.Subset(judg_data, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_split, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_split, batch_size = BATCH_SIZE, shuffle = True)

    #CREATING MODEL
    model = CaseSentimentLSTM(weights_matrix, HIDDEN_SIZE,OUTPUT_SIZE, NUM_LAYERS)
    
    # #CREATE OPTIMIZER AND CRITERION
    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.BCEWithLogitsLoss()

    # #SEND TO GPU
    # model = model.to(device)
    # criterion = criterion.to(device)

    # #TRAIN MODEL
    # train(model)


