from tkinter import HIDDEN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
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

#SETTING THE SEED TO 1 SO NO RANDOM COMPUTATION
torch.manual_seed(0)
random.seed(0)

#SETTING DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Main Tokenizer
tokenizer = get_tokenizer("basic_english")


"""
Summary:The CaseSentimentLSTM is the model used alongside with Sentiment Analysis
in order to predict the favored winner of the case depending on the textdata given.

Model Layout: In order to create a deep learning network, multiple layers are included and
can be viewed in the __init__ method of the model. Each model serves a purpose in order
to help get more accurate results.

Current Completion Progress: Code has been tested up until the Training part. In order to complete the basic
functionality of the model, evaluation as well as testing must be done in order to get back results and to see
results that would be considered normal.

Choices of Constants: Hidden Size is chosen to be 32 for now but can be changed to 16 later to introduce more strict output.
Batch size is chosen to be 32 as a good count for a set of 3303 cases. Num_layers chosen to be 2 in order to introduce some
strictness and stacking into the LSTM model. Output_Size is chosen to be 1 as a default. Train Size and Test Size are at 0.8
and 0.2 as defaults

Roadblocks Encountered Resulting in Delay: Text Data had to be transferred into a custom Dataset in order
to be fed into a DataLoader class. Conflicting tutorials as well as legacy libraries that cannot be used on current desktop.

Functionalities To not be Added: Additional features are not being accounted for and dates will not be separated
due to lack of sufficient data already for training purposes. """


class CaseSentimentLSTM(nn.Module):
    
    def __init__(self, weights_matrix, hidden_size, output_size, num_layers, drop_rate = 0.5):
        super(CaseSentimentLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.word_embeddings, embedding_dim = create_emb_layer(weights_matrix)

        #Initializes the weights_matrix as the weights to be used in the current word_embeddings structure
        self.word_embeddings.weight.data.copy_(torch.tensor(weights_matrix))


        #2 Layered LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first = True, dropout = drop_rate)

        #Dropout Layer to prevent overfit
        self.dropout = nn.Dropout()

        #Fully Connected Linear Layer
        self.fc = nn.Linear(hidden_size, output_size)

        #Final Layer for Fact Positivity given the Party Label
        self.sig = nn.Sigmoid()

    def forward(self, text, text_length):
        
        #Embedding Layer
        embedded = self.dropout(self.word_embeddings(text))
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length.to("cpu"))

        #LSTM Layer
        lstm_out, hidden  = self.lstm(embedded)

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

def binary_accuracy(preds, y):
    """Returns back the accuracy rate of the predictions given"""
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.mean()

    return acc

def train(model: CaseSentimentLSTM , iterator: torch.utils.data.DataLoader, optimizer: optim.Adam, criterion: nn.BCEWithLogitsLoss):
    """A framework function in order to run the test of the model"""

    #Sets the original variables to be passed into the return later on 
    epoch_loss = 0
    epoch_acc = 0

    #Trains the model first on what info has alerady been fed from the word embeddings
    model.train()

    #Goes batch by batch and attempts to make predictions from the text and text_length that is passed to the base nn.Module Class
    for batch in iterator:

        optimizer.zero_grad()
        
        text = batch["text"]
        text_length = batch["text_length"]
        print(text)

        #Prediction and squeezing the predictions into 1 dimension
        preds = model(text,text_length).squeeze(1)

        #Calculating the loss through the preds
        loss = criterion(preds, batch["label"])

        #Calculating the
        acc = binary_accuracy(preds, batch["label"])

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc /len(iterator)

def evaluate(model: CaseSentimentLSTM, iterator: torch.utils.data.DataLoader, criterion: nn.BCEWithLogitsLoss):
    """Framework function in order to evaluate the model its loss and accuracy"""

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
    """A function made in order to take what is in the gloVe files and convert it into a workable dictionary for the code"""
    global glove
    glove = {}

    #Create a Timer to see the load time of the file
    start = time.time()

    #Opens the glove file and starts uploading its vectors(50,100, 200, 300) into a dictionary for Python usage
    with open(GLOVE100D, encoding= "utf-8") as file:
        for line in file:
            dimensions = line.split()
            word = dimensions[0]
            dimensions = np.asarray(dimensions[1:], dtype = "float32")
            glove[word] = dimensions

    end = time.time()
    print("Loading time for GLOVE:", end-start, " seconds")


def check_vocab_instances():
    """Adds in new vocab from the text_data that could not be found in the gloVe file in order to not lead to any errors in the future"""
    vocab = set()
    emb_dim = len(glove["the"])
    dataset, corpus = extraction.extract_data()

    #A for loop over the text corpus to see any unique instances
    for case in corpus:
        for word in case:
            vocab.add(word)

    #Sets the original weights_matrix that will be used later to create the Embedding Layer
    matrix_size = len(vocab)
    weights_matrix = np.zeros((matrix_size, emb_dim))
    words_found = 0

    #Checks for the existence of the key. If it doesn't exist, then add it into the weights matrix with a random vector matrix
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = glove[word]
            words_found +=1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale =0.6, size = (emb_dim,))
    
    return dataset, weights_matrix, words_found

def create_emb_layer(weights_matrix, non_trainable = False):
    """Creates the embedded layer without the initialization of the weights_matrix into the embedding."""
    vocab_size, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(vocab_size, embedding_dim)
    
    if non_trainable:
        emb_layer.weight.requires_grad = False
    
    return emb_layer, embedding_dim

def yield_tokens(iter):
    for text in iter:
        yield tokenizer(text)

def collate_batch(batch):
    text_list, label_list, text_lengths = [],[],[]
    max_vocab_size = 500
    for sample in batch:
        processed_text = vocab(tokenizer((sample["text"])))
        while len(processed_text) < 500:
            processed_text.append(0)
        text_list.append(processed_text)
        text_lengths.append(sample["text_length"])
        label_list.append(sample["label"])
    
    label_list = torch.tensor(label_list, dtype = torch.int64)
    text_lengths = torch.tensor(text_lengths, dtype = torch.int64)
    text_list = torch.tensor(text_list, dtype = torch.int64)
    print(text_list.size())
    return {"label": label_list.to(device), "text":text_list.to(device), "text_length": text_lengths.to(device)}


if __name__ == "__main__":

    #INITIALIZING GLOVE
    initialize_glove()
    dataset, weights_matrix, words_found = check_vocab_instances()

    #CREATING DATASETS
    judg_data = extraction.JudgmentDataset(dataset)
    
    global vocab
    vocab = build_vocab_from_iterator(yield_tokens(judg_data.x_train))

    train_indices, test_indices, _, _ = train_test_split(range(len(judg_data)), judg_data.targets
                                                        , stratify = judg_data.targets, test_size= TEST_SIZE, random_state = 1)
    

    train_split = torch.utils.data.Subset(judg_data, train_indices)
    test_split = torch.utils.data.Subset(judg_data, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_split, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, collate_fn = collate_batch)
    test_loader = torch.utils.data.DataLoader(test_split, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, collate_fn = collate_batch)

    #CREATING MODEL
    model = CaseSentimentLSTM(weights_matrix, HIDDEN_SIZE,OUTPUT_SIZE, NUM_LAYERS)
    
    #CREATE OPTIMIZER AND CRITERION
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    #SEND TO GPU
    model = model.to(device)
    criterion = criterion.to(device)

    #TRAIN MODEL
    train(model, train_loader, optimizer, criterion)


