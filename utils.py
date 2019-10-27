import pickle
import re
import torch


def load_pickle(filename):
    """ Loads a pickled file """
    pickle_load = pickle.load(open(filename, 'rb'))
    return pickle_load

def get_vocab(data):
    """ Gets the vocabulary for the sentences """
    sents = [[x for x in sent] for sent in data]
    vocab = {f: i+1 for i, f in enumerate(sorted(list(set(sum(sents, [])))))}

    return vocab

def encodings(vocab, sentence):
    """"
    Encoding by mapping each word in a sentence to a correspoding index in a vocabulary.
    """
    encoded = []
    for word in sentence:
        try:
            encoded.append(vocab[word])
        except KeyError:
            encoded.append(0)
    encoded_tensor = torch.LongTensor(encoded)

    return encoded_tensor

# 
def normalizeString(s):
    """ 
    Lowercase and remove non-letter characters 
    """
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def indexesFromSentence(voc, sentence, EOS_token):
    """ 
    Takes string sentence, returns sentence of word indexes 
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def split_data(sentences, size=0.2):
    
    test_dict = {}
    train_dict = {}

    # Sentences is a dictionary, keys are integers
    sentence_keys = list(sentences.keys())
    print("Total length of sentences: ".format(len(sentences)))

    # Splitting data into sets
    print("Splitting data into training and tests sets...")
    trainset, testset = train_test_split(sentence_keys, test_size=size)

    # Printing the lengths of the sets
    print("Length of training set: {}".format(len(trainset)))
    print("Length of test set: {}".format(len(testset)))

    # Putting the corresponding values into the new sets as dictionaries
    for k in trainset:
        train_dict[k] = sentences[k]

    for j in testset:
        test_dict[j] = sentences[j]

    return train_dict, test_dict