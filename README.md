# OpenX_ML_Intern
## Intro
The following work is part of recruitment process at OpenX for the position of Machine Learning Intern.
Main task was composed of several steps:
1. Implementing a very simple heuristic that will classify the data (doesn't need to be accurate).
2. Implementing two simple Machine Learning models, based on Scikit-learn library.
3. Implementing a neural network that will classify the data, based on TensorFlow library.
Also creating a function that will find a good set of hyperparameters for the NN 
and plotting training curves for the best hyperparameters.
4. Evaluating models - plots or metrics.
5. Creating a very simple REST API that will serve models; it should allow users to choose a model
and take all necessary input features and return a prediction.
6. (Optional) Creating a Docker container image to serve the REST API.
<br>
Source of data: https://archive.ics.uci.edu/ml/datasets/Covertype  
Data was published by Colorado State University in the USA.  

Tech stack:
- scikit-learn
- tensorflow/keras
- seaborn
- matplotlib
- pandas
- numpy
- flask
- docker
- pickle

To install one of the above:
```
pip install <package_name>
```
## Description
### Dataset
Dataset is being stored as **covtype.data** file, already pre-processed, ready to work with.
Describe types of forest cover - divided into 7 categories. Beyond that there are also 54 columns
describing cartographic variables.
Exact description can be found in **covtype.info** file attached to project, can be
also found at source path.
<br>
<br>
Basically dataset was downloaded and unpacked from archive, then loaded as **pandas** dataframe.
Every model loads data separately from others. In some cases I cut the rows of dataset to reduce time
of compiling and be able to work more quickly and efficiently - dataset contains huge numbers of indexes,
more than 500k (Hardware limitations).
### Heuristic Algorithm

### Two Scikit-Learn classification algorithms 

### TensorFlow neural network

### Evaluation of models

### REST API

### Docker
