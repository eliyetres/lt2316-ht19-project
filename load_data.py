import argparse
import re
import string
from os import listdir
from pickle import dump, load

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def load_doc(filename):
    """ Load stories to memory """
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def split_story(doc):
    """ Split a document into news story and highlights """
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # print(story)
    # print(highlights)
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights


def load_stories(directory):
    """ Load all stories in a directory """
    stories = []
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story': story, 'highlights': highlights})
    return stories


def clean_lines(lines):
    """ Clean list with all lines """
    cleaned = []
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # remove CNN title
        line = line.replace('(CNN) -- ','')
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


def split_data(stories, size=0.33):
    """ Split data into training and test sets, 30% as training data """
    # Splitting data into sets
    print("Splitting data into training and tests sets...")
    trainset, testset = train_test_split(stories, test_size=size)
    # Printing the lengths of the sets
    print("Number of stories saved to training set: {}".format(len(trainset)))
    print("Number of stories saved to test set: {}".format(len(testset)))

    return trainset, testset


def clean_stories(data_path, save_file):
    """ Load stories """
    directory = data_path
    stories = load_stories(directory)
    print("Loaded number of stories: {}.".format(len(stories)))
    # clean stories
    for example in stories:
        example["story"] = clean_lines(example['story'].split('\n'))
        example["highlights"] = clean_lines(example["highlights"])


    # split into training and test sets
    train_dict, test_dict = split_data(stories)
    # saving training and test data to pickle file
    dump(train_dict, open(save_file+'_train.pkl', 'wb'))
    dump(test_dict, open(save_file+'_test.pkl', 'wb'))
    print("Saved stories for training and test to files: {} and {}.".format(
        save_file+"_train.pkl", save_file+"_test.pkl"))
    # save to file
    #dump(stories, open(save_file+'.pkl', 'wb'))
    #print("Saved stories to file: {}.".format(save_file))


# parser = argparse.ArgumentParser(description="Saves a pickled file to disk with the preprocessed texts.")

# parser.add_argument("-path", type=str, dest="data_path", help="Folder name of the CNN stories.")
# parser.add_argument("-save", type=str, dest="save_file", help="Name of the pickled files to be saved to disk. The files will be saved as this name plus train and test, like example_train and example_test")
# args = parser.parse_args()
#clean_stories(args.data_path, args.save_file)
clean_stories("data/cnn/test", "small")


# python load_data.py -path data/cnn/test -save stories_test
