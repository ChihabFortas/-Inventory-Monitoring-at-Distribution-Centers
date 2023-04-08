# Importing packages
import sagemaker
import os
import json
import boto3
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def download_images(files_list, data_path):
	s3_client = boto3.client('s3')
	data_path = os.path.join('dataset',data_path)

	for k, v in files_list.items():
		print("Downloading Images with {} objects to {}", k, data_path)
		directory = os.path.join(data_path ,k)
		if not os.path.exists(directory):
			os.makedirs(directory)
		for file_path in tqdm(v):
			file_name=os.path.basename(file_path).split('.')[0]+'.jpg'
			s3_client.download_file('aft-vbi-pds', os.path.join('bin-images', file_name),os.path.join(directory, file_name))

def download_and_arrange_data():
	with open('file_list.json', 'r') as f:
        	d=json.load(f)
	#spliting data into 65% for traininig, 20% for validationa and 15% for testing
	train = {}
	test = {}
	validation = {}
	for k, v in d.items():
		train[k], test[k] = train_test_split(d[k], test_size=0.35, random_state=0)
		test[k], validation[k] = train_test_split(test[k], test_size=0.60, random_state=0)

	download_images(train, 'train')
	download_images(test, 'test')
	download_images(validation, 'valid')

download_and_arrange_data()
