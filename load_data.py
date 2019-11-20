import argparse
import csv
import re
import string
import sys
from os import listdir
from pickle import dump, load

import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from config import DATA_PATH, EOS_TOKEN, SAVE_TO_PATH, SOS_TOKEN


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

def change_prefixes(word):
    eng_prefixes = {
    "i'm ":"i am ",    
    "he's ":"he is",    
    "she's ":"she is",   
    "you're ":"you are",  
    "we're ":"we are",   
    "they're":"they are",
    "they've":"they have",
    "hadn't": "had not",
    "no.": "number",
    "gov":"governor",
    "it's":"it is",
    "can't":"can not"    
    }
    if word in eng_prefixes.keys():
        word = eng_prefixes[word]
    return word


def remove_alphanum(word):
    word = re.sub(r'\W+', '', word)
    return word


def clean_lines(lines):
    """ Clean list with all lines """
    cleaned = ""
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # remove CNN titles
        line = line.replace('(CNN) -- ', '')
        line = line.replace('(CNN)', '') 
        # replace dash with space to avoid compund words  
        line = line.replace('-', ' ')
        # remove characters
        line = re.sub(r'[\'\?\!\"\*\&\"\:\.\,\(\)]', '', line)
        line = re.sub(r'[Ã©Ã±]', '', line)
        # remove genitive
        line = re.sub(r's', '', line)  
        line = line.replace('  ', ' ')  
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        #line = [w.translate(table) for w in line]        
        # remove tokens with numbers in them
        #line = [word for word in line if word.isalpha()]
        # removing non alphanumeric characters
        #line = [remove_alphanum(w) for w in line]
        line = [change_prefixes(word) for word in line]

        cleaned = cleaned+","+' '.join(line)
    return cleaned


def cut_stories(cleaned, max_len):
    # cut the text at the max length
    cleaned = cleaned[0:max_len]
    cleaned = cleaned.split(",")
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0] 
    # remove the last word because it's probably not a full word
    cleaned = cleaned[:-1]
    cleaned = " ".join(cleaned)
    return cleaned

def cut_highlights(cleaned, max_len):
    cleaned = [c for c in cleaned if len(c) > 0] 
    cleaned = "".join(cleaned)
    # cut the text at the max length
    cleaned = cleaned[0:max_len]
    cleaned = re.sub(r',', ' ', cleaned)
    cleaned = cleaned.split(" ")
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0] 
    # remove the last word because it's probably not a full word
    cleaned = cleaned[:-1]
    cleaned = " ".join(cleaned)
    return cleaned


def split_data(stories, size=0.20):
    # Splitting data into sets
    trainset, testset = train_test_split(stories, test_size=size)
    # Printing the lengths of the sets
    return trainset, testset


def clean_stories(data_path, save_file):
    print("Loading stories...")
    directory = data_path
    stories = load_stories(directory)
    print("Loaded number of stories: {}.".format(len(stories)))
    # clean stories
    for example in stories:
        example["story"] = clean_lines(example['story'].split('\n'))
        example["highlights"] = clean_lines(example["highlights"])
        example["story"] = cut_stories(example['story'], 400)
        example["highlights"] = cut_highlights(example["highlights"], 100)
    # split text
    print("Splitting stories into train and test...")
    train, test = split_data(stories)
    train, val = split_data(train)
    print("Number of stories in training set: {}".format(len(train)))
    print("Number of stories in validation set: {}".format(len(val)))
    print("Number of stories in test set: {}".format(len(test)))

    print("Writing data to files...")
    test_filename = save_file + "_train.csv"
    val_filename = save_file + "_val.csv"
    train_filename = save_file + "_test.csv"

    df1 = pd.DataFrame.from_dict(train)
    df1.to_csv(train_filename, encoding='utf-8', index=False)

    df2 = pd.DataFrame.from_dict(test)
    df2.to_csv(test_filename, encoding='utf-8', index=False)

    df3 = pd.DataFrame.from_dict(val)
    df3.to_csv(val_filename, encoding='utf-8', index=False)

    print("Finished processing data, saved train, validation and test files as {}, {} and {} ".format(
        train_filename, val_filename, test_filename))

    # use this to get one file with all data
    #df4 = pd.DataFrame.from_dict(stories)
    #df4.to_csv("all_stories", encoding='utf-8', index=False) 

# parser = argparse.ArgumentParser(description="Saves a pickled file to disk with the preprocessed texts.")

# parser.add_argument("-path", type=str, dest="data_path", help="Folder name of the CNN stories.")
# parser.add_argument("-save", type=str, dest="save_file", help="Name of the pickled files to be saved to disk. The files will be saved as this name plus train and test, like example_train and example_test")
# args = parser.parse_args()
#clean_stories(args.data_path, args.save_file)
clean_stories(DATA_PATH, SAVE_TO_PATH)


# python load_data.py -path data/cnn/test -save stories_test
