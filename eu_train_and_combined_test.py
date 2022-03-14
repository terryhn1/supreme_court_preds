from cgi import test
import lstm_sentiment_analysis
import torch
import random
import time
from torchtext.legacy import data
import torch.nn as nn
import torch.optim as optim

import io
import os
import torchtext
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader


HIDDEN_SIZE = 16
BATCH_SIZE = 64


tokenizer = lstm_sentiment_analysis.tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":

    #TEXT AND LABELS
    TEXT = data.Field(tokenize = tokenizer, use_vocab = True, lower = True, batch_first = True, include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float, batch_first = True, sequential = False)

    fields = [('index', LABEL),("text", TEXT), ('case_name', TEXT), ("first_party", TEXT), 
    ('second_party',TEXT), ('label', LABEL)]
    
    #validation dataset
    #combined eu and us
    combined_data = data.TabularDataset('datasets/csv_data/combined_data.csv', format = 'csv', fields = fields, skip_header = True)
    print(combined_data.fields)


    TEXT.build_vocab(combined_data, vectors = 'glove.6B.100d', min_freq = 1, unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(combined_data)


    #BUILD SPLITS
    train_split, valid_split = combined_data.split(split_ratio = lstm_sentiment_analysis.TRAIN_SIZE, random_state = random.seed(0))

    train_loader, test_loader = data.BucketIterator.splits((train_split, valid_split), batch_size=32, device = device, sort_key = lambda x: len(x.text), shuffle = True, sort_within_batch = True, sort = False)
    print(train_loader, test_loader)

    #CREATING MODEL
    model = lstm_sentiment_analysis.CaseSentimentLSTM(len(TEXT.vocab), lstm_sentiment_analysis.EMBEDDING_DIM, HIDDEN_SIZE, lstm_sentiment_analysis.OUTPUT_SIZE, lstm_sentiment_analysis.NUM_LAYERS, TEXT.vocab.stoi[TEXT.pad_token])
    model.word_embedding.weight.data.copy_(TEXT.vocab.vectors)

    #Fixing the the unk and pad token
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.word_embedding.weight.data[UNK_IDX] = torch.zeros(lstm_sentiment_analysis.EMBEDDING_DIM)
    model.word_embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]] = torch.zeros(lstm_sentiment_analysis.EMBEDDING_DIM)

    #CREATE OPTIMIZER AND CRITERION
    optimizer = optim.AdamW(model.parameters(), lr = 0.003, weight_decay = 0.3)
    criterion = nn.BCEWithLogitsLoss()

    #SEND TO GPU
    model = model.to(device)
    criterion = criterion.to(device)

    print("Beginning Training")
    #EPOCHS
    best_valid_loss = float('inf')

    for epoch in range(lstm_sentiment_analysis.N_EPOCH):

        train_loader.create_batches()

        start_time  = time.time()
        
        train_loss, train_acc = lstm_sentiment_analysis.train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = lstm_sentiment_analysis.evaluate(model, test_loader, criterion)

        end_time = time.time()

        if valid_loss <  best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'judgement_preds-lstm.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {end_time - start_time}s', flush = True)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.3f}%', flush = True)
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.3f}%', flush = True)