#-*- coding: utf-8 -*-
# author: Raghav Tandon
# Dated: 26.01.2019 
"""
setting up data
"""
import os
import pandas as pd
import json
import logging
import requests
import PIL.Image
import string
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from keras.models import Model
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='folder having data files')
    args = parser.parse_args()
    return args

def read_file(path):
    print("Read file from:" + path)
    file = []
    for line in open(path,"r"):
        file.append(json.loads(line))
    return pd.DataFrame(file)

def get_bal_data(df):
    print("Extracting equal amount of labels from the dataset")
    p1 = df_photo[df.label=="food"][0:2000]
    p1 = p1.append(df[df.label=="drink"][0:2000])
    p1 = p1.append(df[df.label=="inside"][0:2000])
    p1 = p1.append(df[df.label=="outside"][0:2000])
    p1 = p1.append(df[df.label=="menu"][0:2000])
    return(p1)

def update_captions(df1,df2):
    print("Updating the caption field for each photo_id where caption is missing or incorrect")
    for i in df1.index:
        if((df1.caption[i]=="") or ("http" in df1.caption[i])):
            bus_id = df1.business_id[i]
            rest_rating = df2.stars[df2.business_id == bus_id].item()
            if((rest_rating>=1) and (rest_rating<=2)):
                res="one of moderate restaurants"
            elif(((rest_rating>2) or (rest_rating<=4))):
                res="one of good restaurants"
            else:
                res="one of the best restaurants"
                
            if(df1.label[i]=="food"):
                df1.caption[i] = "Relish the food at {r}".format(r=res)
            elif(df1.label[i]=="drink"):
                df1.caption[i] = "Savor drinks at {r}".format(r=res)
            elif(df1.label[i]=="inside"):
                df1.caption[i] = "Check the interiors of {r}, users have rated it {s} stars".format(r=res,s=rest_rating)
            elif(df1.label[i]=="outside"):
                df1.caption[i] = "{r} looks like this from outside, users have rated it {s} stars".format(r=res,s=rest_rating)    
            elif(df1.label[i]=="menu"):
                df1.caption[i] = "Menu from {r}".format(r=res)
    df3 = df1.to_json(orient="records")
    return df3

def feature_extraction(df_json,directory):
    print("converting each photo into pixel format")
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
	# extract features from each photo
    features = dict()
    for d in df_json:
		# load an image from file
        filename = directory+d["photo_id"]+".jpg"
        image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
        image = img_to_array(image)
		# reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
        image = preprocess_input(image)
		# get features
        feature = model.predict(image, verbose=0)
		# get image id
        image_id = d["photo_id"]
		# store feature
        features[image_id] = feature
    return features
                
def load_description(df_json):
    mapping = dict()
    for d in df_json:
        image_id, image_desc = d['photo_id'],d['caption']
        if image_id not in mapping:
            mapping[image_id] = image_desc
    return mapping

def clean_description(descriptions):
    print("Cleaning the text in captions")
    table = str.maketrans('','',string.punctuation)
    for key, desc in descriptions.items():
        desc = desc.split()
        # lower case
        desc = [word.lower() for word in desc]
        # remove punctuation
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # store as string
        descriptions[key] = ' '.join(desc)   

def save_doc(descriptions,filename):
    print("saving descriptions.txt file at:" + filename)
    lines = list()
    for key, desc in descriptions.items():
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    

if __name__ == '__main__':
    args = get_args()
    path = args.path
    photo_path = path+"yelp_dataset/photo.json"
    business_path = path+"yelp_dataset/business.json"
    df_photo = read_file(photo_path)
    df_business = read_file(business_path)
    df_photo_bal = get_bal_data(df_photo)
    df_photo_bal_updated = json.loads(update_captions(df_photo_bal,df_business))
    directory = path+"yelp_photos/photos/"
    features = feature_extraction(df_photo_bal_updated,directory)
    print('Extracted Features: %d' % len(features))
    dump(features, open(path+"prepocess_data/features.pkl", 'wb'))
    descriptions = load_description(df_photo_bal_updated)
    print("loaded : {a}".format(a = len(descriptions)))
    clean_description(descriptions)
    #summarize vocab
    all_tokens = ' '.join(descriptions.values()).split()
    vocabulary = set(all_tokens)
    print('vocab size: {s}'.format(s = len(vocabulary)))
    filename = path+"prepocess_data/descriptions.txt"
    save_doc(descriptions,filename)
    
    
