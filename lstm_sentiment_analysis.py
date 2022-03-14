from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.legacy import data
import torch.optim as optim
import numpy as np
import time
import random
import torch
import torch.nn as nn
import src.extraction as extraction


#CONSTANTS for file reads
GLOVE50D = '.vector_cache/glove.6B.50d.txt'
GLOVE100D = '.vector_cache/glove.6B.100d.txt'
GLOVE200D = '.vector_cache/glove.6B.200d.txt'
GLOVE300D = '.vector_cache/glove.6B.200d.txt'

#CONSTANT FOR HIDDEN DIM/SIZE
HIDDEN_SIZE = 16

#EMBEDDING DIM, CHANGE WHEN USING A DIFFERENT FILE
EMBEDDING_DIM = 100

#BATCH SIZE FOR FORWARD PASSING
BATCH_SIZE = 64

#HOW MANY LSTM MODELS SHOULD BE STACKED
NUM_LAYERS = 2

#OUTPUT SIZE FOR LINEAR MODEL
OUTPUT_SIZE = 1

#TRAINING SIZE
TRAIN_SIZE = 0.7

#TESTING SIZE
TEST_SIZE = 0.2

#EPOCHS FOR TRAINING AND EVALUATION
N_EPOCH = 10

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
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers, pad_idx, drop_rate = 0.5):
        super(CaseSentimentLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        #Initializes the weights_matrix as the weights to be used in the current word_embeddings structure


        #2 Layered LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,  dropout = drop_rate, bidirectional = True)

        #Dropout Layer to prevent overfit
        self.dropout = nn.Dropout(drop_rate)

        #Fully Connected Linear Layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

        #Final Layer for Fact Positivity given the Party Label
        self.sig = nn.Sigmoid()

    def forward(self, text, text_length):
        #Embedding Layer
        embedded = self.dropout(self.word_embedding(torch.transpose(text, 0, 1)))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))


        #LSTM Layer
        lstm_out, (hidden, cell)  = self.lstm(packed_embedded)

        #unpack the lstm output
        out, output_lengths = nn.utils.rnn.pad_packed_sequence(lstm_out)


        #Apply another forward pass then put in the Dropout Layer
        out = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        #Fully Connected Layer
        out = self.fc(out)

        # Return the propagated tensor
        return out

    def init_hidden_state(self, batch_size):
        #creates the h0 layer
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

def binary_accuracy(preds, y):
    """Returns back the accuracy rate of the predictions given"""
    #Sigmoid Layer used here for accuracy predictions
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.mean()

    return acc

def train(model: CaseSentimentLSTM , iterator: torch.utils.data.DataLoader, optimizer: optim.Adam, criterion: nn.BCEWithLogitsLoss):
    """A framework function in order to run the test of the model"""

    #Sets the original variables to be passed into the return later on 
    epoch_loss = 0
    epoch_acc = 0

    #Trains the model first on what info has already been fed from the word embeddings
    model.train()

    #Goes batch by batch and attempts to make predictions from the text and text_length that is passed to the base nn.Module Class
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text

        #Prediction and squeezing the predictions into 1 dimension
        preds = model(text, text_lengths).squeeze(1)

        #Calculating the loss through the preds
        loss = criterion(preds, batch.label)

        #Calculating the
        acc = binary_accuracy(preds, batch.label)


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
            text, text_length = batch.text

            preds = model(text, text_length).squeeze(1)

            loss = criterion(preds, batch.label)

            acc = binary_accuracy(preds, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

if __name__ == "__main__":

    #TEXT AND LABELS
    TEXT = data.Field(tokenize = tokenizer, use_vocab = True, lower = True, batch_first = True, include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float, batch_first = True, sequential = False)

    fields = [('index', LABEL),("text", TEXT), ('case_name', TEXT), ("first_party", TEXT), 
    ('second_party',TEXT), ('label', LABEL)]
    
    judg_data = data.TabularDataset('datasets/csv_data/supreme_court.csv', format = 'csv', fields = fields, skip_header = True)

    TEXT.build_vocab(judg_data, vectors = 'glove.6B.100d', min_freq = 1, unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(judg_data)


    #BUILD SPLITS
    train_split, test_split = judg_data.split(split_ratio = TRAIN_SIZE, random_state = random.seed(0))
    print(len(train_split), len(test_split))

    #train_split, valid_split = judg_data.split(split_ratio = 0.8, random_state = random.seed(0))

    train_loader, test_loader = data.BucketIterator.splits((train_split, test_split), batch_size = BATCH_SIZE, device = device, sort_key = lambda x: len(x.text), shuffle = True, sort_within_batch = True, sort = False)

    #CREATING MODEL
    model = CaseSentimentLSTM(len(TEXT.vocab), EMBEDDING_DIM, HIDDEN_SIZE,OUTPUT_SIZE, NUM_LAYERS, TEXT.vocab.stoi[TEXT.pad_token])
    model.word_embedding.weight.data.copy_(TEXT.vocab.vectors)

    #Fixing the the unk and pad token
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.word_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.word_embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]] = torch.zeros(EMBEDDING_DIM)

    #CREATE OPTIMIZER AND CRITERION
    optimizer = optim.AdamW(model.parameters(), lr = 0.003, weight_decay = 0.3)
    criterion = nn.BCEWithLogitsLoss()

    #SEND TO GPU
    model = model.to(device)
    criterion = criterion.to(device)

    #EPOCHS
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCH):
        start_time  = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion)

        end_time = time.time()

        if valid_loss <  best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'judgement_preds-lstm.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {end_time - start_time}s', flush = True)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.3f}%', flush = True)
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.3f}%', flush = True)