import torch

# DATA_PATH = "data/cnn/test"
# TEST_DATA_PATH = "processed_data/small_test.csv"
# TRAIN_DATA_PATH = "processed_data/small_train.csv"
# SAVE_TO_PATH = "processed_data/small"

DATA_PATH = "data/cnn/test"
TEST_DATA_PATH = "processed_data/small_test.csv"
TRAIN_DATA_PATH = "processed_data/small_train.csv"
SAVE_TO_PATH = "processed_data/small"

# model parameters
EMBEDDING_SIZE = 300 # embedding size needs to be 300 to match the model vectors
HIDDEN_SIZE= 200
BATCH_SIZE = 128
EPOCHS = 4
N_LAYERS = 2  # number of rnn layers (must be same for encoder & decoder)
ENC_DROPOUT = 0.5   	# encoder dropout
DEC_DROPOUT = 0.5   	# decoder dropout (can be different from encoder droput)
CLIP = 5

# Default word tokens
SOS_TOKEN = "<s>"  # Start-of-sentence token
EOS_TOKEN = "<e>"  # End-of-sentence token
PAD_TOKEN = "<pad>"  # Used for padding short sentences
UNK_TOKEN = "<unk>" # this is done in the tokenizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
