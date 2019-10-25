import argparse
import re
import string
from os import listdir
from pickle import dump


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights
 
# load all stories in a directory
def load_stories(directory):
	stories = list()
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)
		# split into story and highlights
		story, highlights = split_story(doc)
		# store
		stories.append({'story':story, 'highlights':highlights})
	return stories
 
# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# strip source cnn office if it exists
		index = line.find('(CNN) -- ')
		if index > -1:
			line = line[index+len('(CNN)'):]
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [w.translate(table) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()] # should numbers be removed?
		# store as string
		cleaned.append(' '.join(line))
	# remove empty strings
	cleaned = [c for c in cleaned if len(c) > 0]
	return cleaned
 
def get_data(data_path,save_file):
    # load stories
	directory = data_path
	stories = load_stories(directory)
	print("Loaded number of stories :{}.".format(len(stories)))
	
	# clean stories
	for example in stories:
		example["story"] = clean_lines(example['story'].split('\n'))
		example["highlights"] = clean_lines(example["highlights"])

		# save to file
		dump(stories, open(save_file+'.pkl', 'wb'))
		print("Saved stories to {}.".format(save_file))

parser = argparse.ArgumentParser(description="Saves a pickled file to disk with the preprocessed texts.")

parser.add_argument("-path", type=str, dest="data_path", help="Folder name of the CNN stories.")
parser.add_argument("-save", type=str, dest="save_file", help="Name of the pickled file to be saved to disk.")

args = parser.parse_args()

get_data(args.data_path, args.save_file)

# python load_data.py -path data/cnn/test -save stories_test
