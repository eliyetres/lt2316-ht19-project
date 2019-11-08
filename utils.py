import pickle
import re
import torch
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from config import EMBEDDING_SIZE,PAD_TOKEN,SOS_TOKEN,EOS_TOKEN
from nltk.tokenize import word_tokenize
from spacy.vectors import Vectors
import spacy



# def get_vocab(data):
#     """ Gets the vocabulary for the sentences """
#     sents = [[x for x in sent] for sent in data]
#     vocab = {f: i+1 for i, f in enumerate(sorted(list(set(sum(sents, [])))))}

#     return vocab

# def encodings(vocab, sentence):
#     """"
#     Encoding by mapping each word in a sentence to a correspoding index in a vocabulary.
#     """
#     encoded = []
#     for word in sentence:
#         try:
#             encoded.append(vocab[word])
#         except KeyError:
#             encoded.append(0)
#     encoded_tensor = torch.LongTensor(encoded)

#     return encoded_tensor


# def split_data(sentences, size=0.33):
	
#     test_dict = {}
#     train_dict = {}

#     # Sentences is a dictionary, keys are integers
#     sentence_keys = list(sentences.keys())
#     print("Total length of sentences: ".format(len(sentences)))

#     # Splitting data into sets
#     print("Splitting data into training and tests sets...")
#     trainset, testset = train_test_split(sentence_keys, test_size=size)

#     # Printing the lengths of the sets
#     print("Length of training set: {}".format(len(trainset)))
#     print("Length of test set: {}".format(len(testset)))

#     # Putting the corresponding values into the new sets as dictionaries
#     for k in trainset:
#         train_dict[k] = sentences[k]

#     for j in testset:
#         test_dict[j] = sentences[j]

#     return train_dict, test_dict

def normalizeString(s):
	""" 
	Lowercase and remove non-letter characters 
	"""
	s = s.lower()
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def indexesFromSentence(voc, sentence):
	""" 
	Takes string sentence, returns sentence of word indexes 
	"""
	return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]


def pad_vec_sequences(sequences, maxlen=40):
	new_sequences = []
	for sequence in sequences:

		orig_len, vec_len = np.shape(sequence)

		if orig_len < maxlen:
			new = np.zeros((maxlen, vec_len))
			for k in range(maxlen-orig_len,maxlen):
				new[k:, :] = sequence[k-maxlen+orig_len]

		else:
			new = np.zeros((maxlen, vec_len))
			for k in range(0,maxlen):
				new[k:,:] = sequence[k]

		new_sequences.append(new)

	return np.array(new_sequences)


def add_tags(stories):
	X_stories = []
	y_highlights = []
	for story in stories:
		new_story=""
		new_highlight=""
		for article in story["story"]:
			new_article = SOS_TOKEN+" "+article+" "+EOS_TOKEN
			new_story+=" "+new_article
		new_story=new_story.split()
		for summary in  story["highlights"]:
				new_summary = SOS_TOKEN+" "+summary+" "+EOS_TOKEN
				new_highlight+=" "+new_summary
		new_highlight=new_highlight.split()
		X_stories.append(new_story)
		y_highlights.append(new_highlight)   	

	return X_stories, y_highlights	



def load_model(model_path):
	model = KeyedVectors.load_word2vec_format(model_path, binary=True)
	return model

def cut_parts(text,n):
	result = []
	for sentence in text:
		if n <= 0:
			return result
		sli = sentence
		if len(sli) > n:
			sli = sli[:n]
			result.append(sli)  
		else:
			result.append(sli)     
		n -= len(sli)
	return result



def get_word_embedding(token, model):
	# generate an array of zeros of length = embedding_size
	if token in ["<unk>",PAD_TOKEN]:
		return np.zeros(EMBEDDING_SIZE, dtype=np.float32)
	# Assign random vector to <s>, </s> token
	elif token in [SOS_TOKEN,EOS_TOKEN]:    
		return np.random.normal(0, 1, EMBEDDING_SIZE)
	try:
		return model[token]
	except KeyError:
		# generate an array of zeros of length = embedding_size
		print("{} not found in model!".format(token))        
		return np.zeros(EMBEDDING_SIZE, dtype=np.float32)



def shorten_data(stories,max_story=400,max_highlight=100):
    highlight_lengths = []
	story_lengths = []
	for example in stories:    
		st_len = []
		hi_len = []
		highlight = example["highlights"]
		st = example["story"]
		
		highlight = cut_parts(highlight,max_highlight)  
		story = cut_parts(st,max_story)
		example["highlights"]=highlight
		example["story"]=story
		[st_len.append(len(y)) for y in story]           
		[hi_len.append(len(y)) for y in highlight]
		story_lengths.append(sum(st_len))
		highlight_lengths.append(sum(hi_len))

		highlight_avg = sum(highlight_lengths)/len(highlight_lengths)
		story_avg = sum(story_lengths)/len(story_lengths)
		print("Highlights average character length: {}".format(highlight_avg))
		print("Story average character length: {}".format(story_avg))

		return stories	