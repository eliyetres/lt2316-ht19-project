import argparse
import os
import pickle
import sys
import time
from time import gmtime, strftime

import torch
from utils import pickle_load
from dataloader import Dataset
from torch.utils import data
from encoder_RNN import EncoderRNN
from decoder_RNN import LuongAttnDecoderRNN

# load from file
stories = pickle_load(open('cnn_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))

num_words = stories.count()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 200


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
# Load trained model params
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()
print('Models built and ready to go!')