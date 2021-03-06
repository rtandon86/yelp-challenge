#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:27:23 2019

@author: raghav
"""

import os
import pandas as pd
import json
import logging
import requests
import PIL.Image
import string
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
    # Fine tuning- decreasing the representational capacity of the sequence encoder(smaller sized fixed-length vector)
    # Fine tuning- fe2 = Dense(128, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    # Fine tuning- updating the represntation capacaity of word embedding by changing dimension
    # Fine tuning- se1 = Embedding(vocab_size, 128, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
    # Fine tuning- decreasing the representational capacity of the sequence encoder (smaller sized fixed-length vector)
    # Fine tuning- se3 = LSTM(128)(se2)
    # Fine tuning- double the number of LSTM layers 
    # Fine tuning- se4 = LSTM(256)(se3)
	# decoder model
	decoder1 = add([fe2, se3])
    # Fine tuning- adding language model()
	# Fine tuning- lm1 = LSTM(256)(decoder1)
	# Fine tuning- decoder2 = Dense(256, activation='relu')(lm1)
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

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
    train_features = load_photo_features(business_path, train_dataset)
    print('Photos: train=%d' % len(train_features))
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    max_length = max_length(train_descriptions)
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
    #loading validate data
    val_descriptions = load_clean_descriptions(file_path, val_dataset)
    val_features = load_photo_features(business_path, val_dataset)
    print('Photos: validate=%d' % len(val_features))
    X1val, X2val, yval = create_sequences(tokenizer, max_length, val_descriptions, val_features)
    
    
    # loading test data 
    test_descriptions = load_clean_descriptions(file_path, test_dataset)
    test_features = load_photo_features(business_path, test_dataset)
    print('Photos: test=%d' % len(test_features))
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)
    
    #executing model
    model = define_model(vocab_size, max_length)
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit([X1train, X2train], ytrain, epochs=7, verbose=2, callbacks=[checkpoint],validation_data=([X1val,X2val],yval))
