import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from pickle import dump
from pickle import load as pload
import numpy as np
import spacy
import torch
import math

import torch.nn as nn
from torch import random
from torch.optim import Adam
from torch.utils import data
from dataloader import Dataset
import torchtext
from torchtext.data import Field, Iterator, TabularDataset, BucketIterator,interleave_keys

from LSTM_encoder_decoder import Encoder,Decoder,Seq2Seq


from GRU_encoder_decoder import Encoder as GRUencoder
from GRU_encoder_decoder import Decoder as GRUdecoder
from GRU_encoder_decoder import Attention as GRUattention

from config import BATCH_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, EPOCHS, DEVICE, N_LAYERS, LEARNING_RATE, SOS_TOKEN, EOS_TOKEN

# set up cuda
SEED = 23
random.seed()
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
spacy.prefer_gpu()
device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

# for tokenizing the english sentences
spacy_en = spacy.load('en_core_web_sm')


def tokenize_en(text):
	# tokenizes the english text into a list of strings(tokens)
	return [tok.text for tok in spacy_en.tokenizer(text)]

#sequential=True specify that this column holds sequences.
SRC = Field(sequential=True,include_lengths=False,tokenize=tokenize_en,  init_token=SOS_TOKEN, eos_token=EOS_TOKEN)
TRG = Field(sequential=True,include_lengths=False,tokenize=tokenize_en, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

fields = [('stories', SRC), ('highlights', TRG)]
train = TabularDataset(
	path='processed_data/train.csv', format='csv', fields=fields, skip_header=True)


# build vocabulary, not include words less frequent that 2 times
SRC.build_vocab(train, min_freq=1, max_size=15000, vectors="glove.6B.300d")
TRG.build_vocab(train, min_freq=1, max_size=15000, vectors="glove.6B.300d")

print("Vocab built.")
print(f"Vocabulary Size: {len(SRC.vocab)}")


# BucketIterator returns a Batch object instead of text index and labels. Also Batch object is not iterable like pytorch Dataloader. A single Batch object contains the data of one batch .The text and labels can be accessed via column names.
iterator = BucketIterator(train, sort_key=lambda x:(len(x.stories), len(x.highlights)), batch_size=BATCH_SIZE, device=DEVICE, sort_within_batch=True, train=True, shuffle=True)


print("train.fields:", train.fields)
print()
print(vars(train))  # prints the object

vocab = SRC.vocab
emb_dim = EMBEDDING_SIZE
embed = nn.Embedding(len(vocab), emb_dim)
embed.weight.data.copy_(vocab.vectors)
pretrained_vec = vocab.vectors

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
#OUTPUT_DIM = 200
ENC_EMB_DIM = 300   # encoder embedding size
DEC_EMB_DIM = 300   # decoder embedding size (can be different from encoder embedding size)
HID_DIM = 200       # hidden dimension (must be same for encoder & decoder)
N_LAYERS = 2        # number of rnn layers (must be same for encoder & decoder)
ENC_DROPOUT = 0.5   # encoder dropout
DEC_DROPOUT = 0.5   # decoder dropout (can be different from encoder droput)
CLIP = 5 			# gradient clip value


print("input dim: ", INPUT_DIM)
print("output dim: ", OUTPUT_DIM)
print ('Embedding layer is ', embed)
print ('Embedding layer weights ', embed.weight.shape)



encoder = Encoder(device,INPUT_DIM, ENC_EMB_DIM, pretrained_vec, 200, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(device,DEC_EMB_DIM, pretrained_vec, 200,OUTPUT_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(device,encoder, decoder)


pad_idx = TRG.vocab.stoi['<pad>']
print(pad_idx)
# # set model parameters
# loss function calculates the average loss per token
# passing the <pad> token to ignore_idx argument, will ignore loss whenever the target token is <pad>
criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=pad_idx)
optimizer = torch.optim.Adam(encoder.parameters(), LEARNING_RATE)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
model=model.to(device)
criterion.to(device)

encoder.train()
decoder.train()

def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.normal_(param.data, mean=0, std=0.01)
		
model.apply(init_weights)



batch = next(iter(iterator))
#print(batch.stories)
print(batch.stories[0].size())
print(batch.stories[1].size())
print("--------------")
#print(batch.highlights)
print(batch.highlights[0].size())
print(batch.highlights[1].size())

def train_model(model, iterator, optimizer, criterion, clip):
	model.train()
	epoch_loss = 0.0
	for i, batch in enumerate(iterator):
		src  = batch.stories
		trg  = batch.highlights
		src,trg=src.to(device),trg.to(device)
		

		optimizer.zero_grad()

		# trg is of shape [sequence_len, batch_size]
		# output is of shape [sequence_len, batch_size, output_dim]
		output = model(src, trg)

		# loss function works only 2d logits, 1d targets
		# so flatten the trg, output tensors. Ignore the <sos> token
		# trg shape shape should be [(sequence_len - 1) * batch_size]
		# output shape should be [(sequence_len - 1) * batch_size, output_dim]
		

		#output = output[1:].view(-1, output.shape[2])
		#trg = trg[1:].view(-1)

		output = output[1:].view(-1, output.shape[-1])
		trg = trg[1:].view(-1)


		output, trg = output.to(device), trg.to(device)
		loss = criterion(output, trg)

		# backward pass
		loss.backward()

		# clip the gradients to prevent them from exploding (a common issue in RNNs)
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

		# update the parameters
		optimizer.step()

		epoch_loss += loss.item()

	# return the average loss
	return epoch_loss / len(iterator)





def evaluate(model, iterator, criterion):
	''' Evaluation loop for the model to evaluate.
	Args:
		model: A Seq2Seq model instance.
		iterator: A DataIterator to read the data.
		criterion: loss criterion.
	Returns:
		epoch_loss: Average loss of the epoch.
	'''
	#  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
	model.eval()
	# loss
	epoch_loss = 0

	# we don't need to update the model parameters. only forward pass.
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			src  = batch.stories
			trg  = batch.highlights
			src,trg=src.to(device),trg.to(device)

			output = model(src, trg, 0)     # turn off the teacher forcing

			# loss function works only 2d logits, 1d targets
			# so flatten the trg, output tensors. Ignore the <sos> token
			# trg shape shape should be [(sequence_len - 1) * batch_size]
			
			# output shape should be [(sequence_len - 1) * batch_size, output_dim]

			#output  = output[1:].view(-1, output.shape[2])
			output = output[1:].view(-1, output.shape[-1])
			trg = trg[1:].view(-1)

			output,trg = output.to(device), trg.to(device)
			loss = criterion(output, trg)

			epoch_loss += loss.item()
	return epoch_loss / len(iterator)



def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

print("Evaluating model...")
test = TabularDataset(path='processed_data/test.csv', format='csv', fields=fields, skip_header=True)
test_iterator = BucketIterator(test, sort_key=lambda x:(len(x.stories), len(x.highlights)), batch_size=1, device=DEVICE, sort_within_batch=True, train=False, shuffle=False)

test_loss = evaluate(model, iterator, criterion)
print(test_loss)


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
	
	start_time = time.time()
	
	train_loss = train_model(model, iterator, optimizer, criterion, CLIP)
	valid_loss = evaluate(model, test_iterator, criterion)
	
	end_time = time.time()
	
	epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	
	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), 'tut1-model.pt')
	
	print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
	print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
	print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')