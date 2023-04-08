# Inventory Monitoring at Distribution Centers

## Udacity AWS Machine Learning Engineer - Capstone Project
This is a Deep learning project done as part of Udacity's capstone project of AWS Machine Learning Engineer Nanodegree Program.

Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. Occasionally, items are misplaced while being handled, so the contents of some bin images may not match the recorded inventory of that bin.

Now, this project is about building a model that can count the number of objects in each bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items

The solution here is to use AWS SageMaker and good machine-learning engineering practices to fetch data from Amazon Bin Image Dataset, preprocess it, and then train a pre-trained model that can classify the image based on the number of objects in the bin

## Prerequisites
1. Amazon Web services Account with valid Billing status 
2. This repo code as it heavily based on Amazon Bin Image Dataset(ABID) Challenge and Udacity project starter template codes

## Project Set Up and Installation
To start this project:

1. you need to log in to AWS console, to SageMaker Studio and Create a Notebook instance, use 'ml.t3.medium' for now as just we need to run code with no heavy tasks
![1.PNG](/Screenshots/Training/0.PNG)

2. Open the notebook and clone this repo

3. You need to setup an S3 Bucket for this Project
![3.PNG](/Screenshots/Training/3.PNG)

4. open [sagemaker.ipynb](sagemaker.ipynb) and start running the code cells.

## Dataset

### Overview
the dataset provided to us by Amazon warehouses that contain approximately 500,000 image of bins. In our project, we will only consider a small part of this dataset, about 10,441 images split between training, validation, and testing, in order to evaluate the performance of the model and then launch a large training job on the whole dataset.

### Access
The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. You can download and find the details at [here](https://registry.opendata.aws/amazon-bin-imagery/).

### Preparing Data
1. To build this project you have to use the [Amazon Bin Images Dataset](https://registry.opendata.aws/amazon-bin-imagery/)

2. Download the dataset: You are encouraged to use only subset of the dataset to prevent any excess SageMaker credit usage since this is a large dataset, the code to download a small subset of that data is provided.

3. Preprocessing and cleaning the dataset files

4. Then Upload them to an S3 bucket so that SageMaker can use them for training

## Model Training
We choose the ResNet model, which is short for Residual Network, is one of the best-performing models for computer vision tasks. To refine our model, we will start with a few more steps:

1. Install necessary dependencies

2. Read and Preprocess data: Before training the model, we need to read, load and preprocess the training, testing and validation data

3. Create and load the pre-trained model from the framework database or from Amazon Bin Image Dataset(ABID) Challenge, after that Add fully connected layers

4. Create a Loss and optimization Function.

5. Setup the training estimator: Submit a job to train the Model and log any important metrics for debugging and profilling 

![Hyperparameter Tuning Job](/Screenshots/Training/1.PNG)  

![Traning Job](/Screenshots/Training/2.PNG) 

6. Evaluate the model using the validation dataset.  


## Machine Learning Pipeline
1. Upload Training Datato the S3 bucket.

2. Write a Model Training Script to Tune, train, and Inference the model on that dataset.

3. Use SageMaker to run that Tuning and training script to train the model.


## Standout Suggestions

### Model Deployment:
Once you have trained your model, you can deploy it using SageMaker endpoint. The endpoint can be queried using the code in the notebook. The test dataset is already in workspace:

1. Fetch the image.
2. Preprocess the image
3. Send it to endpoint for prediction.
4. Compare the result with the Ground Truth.

### Hyperparameter Tuning: 
To improve the performance of the model, we used previously SageMaker’s Hyperparameter Tuning to search through a hyperparameter space and get the value of the best hyperparameters, using the [hpo.py](code/hpo.py) file, as a result we the tuning job returned this set of hyperparameters:
``{'batch_size': 128, 'learning_rate': '0.06246976097402943', 'epochs': '11'}``

### Multi-Instance Training: 
we going to train the same model but this time distributing the training workload across multiple instances for faster training.

### Reduce Costs: 

To reduce the cost of your machine learning engineering pipeline, we did a cost analysis and then use spot instances to train your model.

Training on ``ml.g4dn.xlarge``, is a GPU optimized instance but it cost is too high compared to CPU optimized intances like ``ml.t3.medium``, our Training Job ook around 20 minutes in 25 epochs, which did cost us around $0.25 for 20 minutes, but for ``ml.t3.medium`` with the same amount will run for 5 Hours, that much of the cost reduction we need, especially using spo instances.   

![4.PNG](/Screenshots/Training/4.PNG)  

so we start using EC2 For training:

![1.PNG](/Screenshots/ec2/1.PNG)  

![2.PNG](/Screenshots/ec2/2.jpg)  

![3.PNG](/Screenshots/ec2/3.PNG)  

![4.PNG](/Screenshots/ec2/4.PNG)  

![5.PNG](/Screenshots/ec2/5.PNG)  

Then we clone this repo and run these commmand lines to install neccesary packages:

```
pip install smdebug torch torchvision tqdm ipywidgets bokeh
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install easydev colormap colorgram.py extcolors
```
then run the ``python data.py`` to download the dataset, after that launch the training job with ``python code/ec2train.py`` after modifying the args parameters in the file, and let the model train for a long time.