#!/bin/bash

#download kaggle data to this current directory before proceed!

if [ -f "fer2013.tar.gz" ]
then
     	echo "Data check pass"
	
	#clean directory
	rm -rf biteam dataset interpreters

	#package
	pip install -r requirements.txt

	#directories
	mkdir -p biteam dataset/kaggle interpreters

	#xception
	git clone https://github.com/oarriaga/face_classification.git biteam

	#lime
	git clone https://github.com/marcotcr/lime.git interpreters
	cp -rf src/lime_image.py interpreters/lime/lime_image.py 

	#data tar
	tar -xzf fer2013.tar.gz -C ./dataset/kaggle/
	tar -xzf fer2013.tar.gz -C ./biteam/datasets/

	#images gen script
	wget -O ./dataset/kaggle/fer2013/gen_record.py "https://kaggle2.blob.core.windows.net/forum-message-attachments/179911/6422/gen_record.py"
	echo "Please wait... generating image file"
	python ./dataset/kaggle/fer2013/gen_record.py

	#sample trained model
	cp -rf ./fer2013_mini_XCEPTION.hdf5 ./biteam/trained_models/emotion_models/fer2013_mini_XCEPTION.hdf5

        #complete
	echo "Project successfully installed"
	
else
	echo "No tar.gz file found"
        echo "Please download the data first and put it in this current directory"
	echo "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data"
fi



