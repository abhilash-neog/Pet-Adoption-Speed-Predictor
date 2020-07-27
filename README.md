# Pet Adoption Speed Predictor

A Data Science capstone project completed in fulfilment of the Coursera course - IBM Advanced Data Science Capstone - the final course in the IBM Advanced Data Science Specialization.

## Dataset

Kaggle dataset - [PetAdoption](https://www.kaggle.com/c/petfinder-adoption-prediction/data) has been used. Dataset contains tabular, image and text (descriptive) data, which makes it quite challenging and interesting to work with. In this project, only the tabular and image data has been used to train and validate the model. More information about the dataset, its attributes and image data can be found in the above link.

## Notebook

The major(primary) tasks present within the notebook are:
* ETL (Extract Transform Load)
* Data Quality Assessment
* Data Exploration
* Data Visualization
* Feature Engineering
* Model Definition
* Model training
* Model Evaluation

## Proposed Model

The proposed model comprises of 3 Neural networks, one of which is a pretrained network (on the imagenet dataset). This 3-net model has been approached due to the presence of categorical + continuous data (including images). 
* The pretrained network ([DenseNet169](https://keras.io/api/applications/densenet/) has been used to obtain image features (*feature_vec1*) from the pet images. 
* The 1st network (NN-1) is trained on the tabular/relational dataset (all the attributes after feature engineering are categorical) with the actual labels.
* Trained NN-1 output is then used to prepare the input for the 2nd network (NN-2). An 1-D feature vector (*feature_vec2*) is extracted from a Dense layer of the trained NN-1.
* Both feature vectors (*feature_vec1* and *feature_vec2* - both 1D) are then concatenated to form a new 1-D feature vector which forms the input for training NN-2
* The architecture can be imagined to be something similar to the following (except the pretrained network is missing which takes in the continous data):

[Image taken from StackExchange](https://datascience.stackexchange.com/questions/29634/how-to-combine-categorical-and-continuous-input-features-for-neural-network-trai)
![alt text](https://i.stack.imgur.com/QgQFq.png)

## Technology

Keras(latest version) with Tensorflow 2.2 has been used in the project. Python 3.7.x has been used to programme the solution. Should work with any python version >= 3.5. Training the 1st network (NN-1) can be done on CPU, whereas the NN-2 training require GPU computation (TPU recommended), as batches of data (tabular + image) are generated dynamically (preprocessing of images done on the generated batch before feeding into the net) during training. 

## Additional

* The ADD (Architectural Decisions Document) can be found in the repo along with the notebook
* A gist of the notebook - [notebook gist](https://gist.github.com/abhilash97/11945d1cdfe5658432d59932f1baeb88)
* Project Presentation (with a demo of the notebook) can be found [here](https://www.youtube.com/watch?v=iqC-eZujNwE)


