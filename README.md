# Yelp-challenge

Image Captioning 
------------
Using this model, when the user provides a image, the model generates a generic caption to indicate what is present in the image.

Data and Environment Set Up 
------------
Please download the datasets available at http://www.yelp.com/dataset_challenge , you will receive two .tar files with below descriptions -
yelp_photos.tar
yelp_dataset.tar

Extract the above files in the folder, doing so you will have two folders, one from each tar file.

To ensure the code runs successfully please install all the packages mentioned in requirements.txt file.

Code Set Up
------------
You will find two files -
1. preprocess_data.py - this code extracts some of the images and data from photos and business file to be used in this project, also it ensures that all the captions are valid captions if not the code creates caption for the image. All the images are sent to feature extraction module, where using VGG model all relevant features are extracted. All the captions also undergo text cleaning.

2. train.py - This code splits the data in train and validate set defines the model and trains the model.

3. evaluate.py - This code evalutes the trained model on the test data.

Code Execution
------------
If you have downloaded the yelp file in downloads and have extracted the files from .tar files in downloads then -
to run preprocess_data.py:
	python preprocess_data.py --path /Users/owner/Downloads/
to run train.py:
	python train.py --path /Users/owner/Downloads/
Note: --path should be followed by the path where folders yelp_photos and yelp_dataset are present.
