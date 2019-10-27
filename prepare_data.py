from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from utils import load_pickle
from pickle import load,dump
import torch
import numpy as np

# embedding size needs to be 300 to match the model vectors
embedding_size = 300


def load_model(model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


def get_word_embedding(token, model):
    #not_found = []

    # generate an array of zeros of length = embedding_size
    if token in ["<UNK>","<PAD>"]:
        return np.zeros(embedding_size, dtype=np.float32)
    # Assign random vector to <s>, </s> token
    elif token in ["<EOS>","<GO>"]:    
        return np.random.normal(0, 1, embedding_size)
    try:
        return model[token]
    except KeyError:
        # generate an array of zeros of length = embedding_size
        print("{} not found in model!".format(token))        
        return np.zeros(embedding_size, dtype=np.float32)
        # check all words not found in the model
        # if token not in not_found:
        #     not_found.append(token)

def get_sent_embedding(sent, model):
    sent_embedding = []
    tokens = word_tokenize(sent)
    for token in tokens:
        token_embedding = get_word_embedding(token, model)
        sent_embedding.append(token_embedding)
    return torch.Tensor(sent_embedding)


def tokenize_stories(stories):
    """ 
    Tokenizes the stories and highlights of the training set.
    Returns new lists split on individual words. 
    """
    words = set()
    for example in stories:
        tokenized_example =[]
        # tokenize sentences
        tokenized_example.append([word_tokenize(sentence) for sentence in example["story"]])
        # append words to vocabulary
        [words.update(token) for sublist in tokenized_example for token in sublist] 
        example["story"] = tokenized_example

        tokenized_example =[]
        [words.update(token) for sublist in tokenized_example for token in sublist] 
        example["highlights"] = tokenized_example

    # Special tokens that will be added to our vocab
    # Unknown, padding, start, end
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]  
    #create vocabulary of words mapped to integers
    vocab = dict(enumerate(codes+list(words)))

    return vocab






# load from file
stories = load(open('small_test.pkl', 'rb'))
print("Loaded stories: {}.".format(len(stories)))




vocab = tokenize_stories(stories)
vocab_size = len(vocab)+1
dump(vocab, open('vocab.pkl', 'wb'))
print("Vocabulary Size: {}".format(vocab_size))
print("Saved vocabulary to {}.".format("vocab.pkl"))


#we would create a reversed dict  
reversed_dict = dict(zip(vocab.values(), vocab.keys()))#then we would simply for the 2 cases (training , or validation)

#define a max len for article and for the summary  
article_max_len = 50
summary_max_len = 15




model = load_model("../lt2212-v19-a4/data/GoogleNews-vectors-negative300.bin")
for token in vocab:
    get_word_embedding(token, model)




