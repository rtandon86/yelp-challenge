#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:53:54 2019

@author: raghav
"""

import os
import pandas as pd
import json
import logging
import requests
import PIL.Image
import string
from numpy import argmax
from numpy import array
from pickle import load

# split file into train and validate
def split_file(filename):
    #open file:
    file = open(filename,'r')
    text = file.read()
    file.close()
    train_dataset = list()
    val_dataset = list()
    test_dataset = list()
    ctr = 0
    for line in text.split('\n'):
        identifier = line.split(' ')[0]
        ctr = ctr + 1
        if((ctr >= 0 and ctr <= 400) or (ctr >= 600 and ctr <= 1000) or(ctr >= 1200 and ctr <= 1600) or(ctr >= 1800 and ctr <= 2200) or(ctr >= 2400 and ctr <= 2800)):
            train_dataset.append(identifier)
        elif((ctr > 400 and ctr <= 500) or (ctr > 1000 and ctr <= 1100) or(ctr > 1600 and ctr <= 1700) or(ctr > 2200 and ctr <= 2300) or(ctr > 2800 and ctr <= 2900)):
            val_dataset.append(identifier)
        else:
            test_dataset.append(identifier)
    return train_dataset, val_dataset, test_dataset


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        identifier = line.split(' ')[0]
        dataset.append(identifier)
    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions


# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text



# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
		# generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        print("yhat: {a}".format(a=yhat))
        print("acual: {a}".format(a=desc_list))
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
 
    
def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='folder having data files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    path = args.path
    file_path = path+"preprocess_data/descriptions.txt"
    business_path = path+"preprocess_data/features.pkl"
    train_dataset, val_dataset, test_dataset = split_file(file_path)
    print('Train Dataset: %d' % len(train_dataset))
    train_descriptions = load_clean_descriptions(file_path, train_dataset)
    print('Descriptions: train=%d' % len(train_descriptions))
    #train_features = load_photo_features(business_path, train_dataset)
    #print('Photos: train=%d' % len(train_features))
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)
    #X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
    #loading test data
    # descriptions
    test_descriptions = load_clean_descriptions(file_path, test_dataset)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = load_photo_features(business_path, test_dataset)
    print('Photos: test=%d' % len(test_features))
    #X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)
    # load the model
    filename = 'model-ep002-loss1.940-val_loss1.699.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)