import argparse
import math
import os
import pickle
import sys
import time
import dill
from datetime import datetime
#from pickle import dump
from pickle import load as pload

import numpy as np
import spacy
import torch
import torch.nn as nn
import torchtext
from torch import random
from torch.optim import Adam
from torch.utils import data
from torchtext.data import BucketIterator, Field, TabularDataset

import config
from GRU_encoder_decoder import GRUAttention, GRUDecoder, GRUEncoder, Seq2Seq

from utils import epoch_time,tokenize_en,evaluate

# set up cuda
SEED = 23
random.seed()
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
spacy.prefer_gpu()

#sequential=True specify that this column holds sequences.
SRC = Field(sequential=True,include_lengths=False,tokenize=tokenize_en,  init_token=config.SOS_TOKEN, eos_token=config.EOS_TOKEN)
TRG = Field(sequential=True,include_lengths=False,tokenize=tokenize_en, init_token=config.SOS_TOKEN, eos_token=config.EOS_TOKEN)
# fields are columns in csv file
# stories = source
# highlights = target
fields = [('stories', SRC), ('highlights', TRG)]
train_data = TabularDataset(
	path=config.TRAIN_DATA_PATH, format='csv', fields=fields, skip_header=True)

# build vocabulary: exclude non-frequest words, limit vocab size
SRC.build_vocab(train_data, min_freq=3, max_size=20000, vectors="glove.6B.300d")
TRG.build_vocab(train_data, min_freq=3, max_size=20000, vectors="glove.6B.300d")
print("Vocab built.")
print(f"Source vocabulary size: {len(SRC.vocab)}")
print(f"Target vocabulary size: {len(TRG.vocab)}")
print("10 most frequent words in the vocab")
print(SRC.vocab.freqs.most_common(10))


torch.save(SRC,'SRC.pt')
torch.save(TRG,'TRG.pt')

# BucketIterator returns a Batch object instead of text index and labels. Also Batch object is not iterable like pytorch Dataloader. A single Batch object contains the data of one batch.The text and labels can be accessed via column names.
train_iterator = BucketIterator(train_data, sort_key=lambda x:(len(x.stories), len(x.highlights)), batch_size=config.BATCH_SIZE, device=config.DEVICE, sort_within_batch=True, train=True, shuffle=True)


print("train.fields:", train_data.fields) # should be stories and highlights
#print()
#print(vars(train))  # prints the object

vocab = SRC.vocab
emb_dim = config.EMBEDDING_SIZE
embed = nn.Embedding(len(vocab), emb_dim)
embed.weight.data.copy_(vocab.vectors)
pretrained_vec = vocab.vectors
pad_idx = SRC.vocab.stoi['<pad>']
#print(pad_idx)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

print("input dim: ",INPUT_DIM)
print("output dim: ",OUTPUT_DIM)
print ('Embedding dim is', embed)
print ('Embedding dim weights', embed.weight.shape)

attention = GRUAttention(
    config.HIDDEN_SIZE, 
    config.HIDDEN_SIZE)
encoder = GRUEncoder(
    INPUT_DIM, config.HIDDEN_SIZE, 
    config.HIDDEN_SIZE, config.HIDDEN_SIZE, 
    config.ENC_DROPOUT)
decoder = GRUDecoder(
    OUTPUT_DIM, 
    config.HIDDEN_SIZE, 
    config.HIDDEN_SIZE, 
    config.HIDDEN_SIZE, 
    config.DEC_DROPOUT, 
    attention)
model = Seq2Seq(
    encoder, decoder,
    config.PAD_TOKEN, 
    config.SOS_TOKEN, 
    config.EOS_TOKEN, 
    config.DEVICE).to(config.DEVICE)


# # set model parameters
# loss function calculates the average loss per token
# passing the <pad> token to ignore_idx argument, will ignore loss whenever the target token is <pad>
criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=pad_idx)
#optimizer = torch.optim.Adam(encoder.parameters(), LEARNING_RATE)
optimizer = torch.optim.Adam(model.parameters(), config.LEARNING_RATE)
# Use appropriate DEVICE
encoder = encoder.to(config.DEVICE)
decoder = decoder.to(config.DEVICE)
model=model.to(config.DEVICE)
criterion.to(config.DEVICE)

encoder.train()
decoder.train()



#### Test
fields = [('stories', SRC), ('highlights', TRG)]
train_data = TabularDataset(
	path=config.TRAIN_DATA_PATH, format='csv', fields=fields, skip_header=True)

test_data = TabularDataset(
	path=config.TEST_DATA_PATH, format='csv', fields=fields, skip_header=True)
test_iterator = BucketIterator(test_data, sort_key=lambda x:(len(x.stories), len(x.highlights)), batch_size=1, device=config.DEVICE, sort_within_batch=True, train=True, shuffle=True)

criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=pad_idx)



def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)


def train(model, iterator, optimizer, criterion, clip):    
    model.train()    
    epoch_loss = 0    
    for i, batch in enumerate(iterator):        
        src = batch.stories
        trg = batch.highlights

        src_len=len(src)
        
        optimizer.zero_grad()

        print(src)
        print(src.size()) 
        print(src.type(), type(src))
        
        
        #output = model(src, trg)
        output, attention = model(src, src_len, trg)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)        
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        
        optimizer.step()        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


best_valid_loss = float('inf')

for epoch in range(config.EPOCHS):    
    start_time = time.time()    
    train_loss = train(model, train_iterator, optimizer, criterion, config.CLIP)
    valid_loss = evaluate(model, test_iterator, criterion)    
    end_time = time.time()    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        #torch.save(model.state_dict(), 'tut3-model.pt')
        torch.save(model, 'model-GRU.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')