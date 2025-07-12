##Predicting Stars, Galaxies & Quasars with Machine Learning
This repository contains a machine learning project focused on classifying astronomical objects (Stars, Galaxies, and Quasars) based on their photometric and spectral features. The project demonstrates a complete workflow from data preprocessing and model training to evaluation and individual sample inference.

##Project Description
The primary goal of this project is to build and evaluate various machine learning models capable of accurately classifying celestial objects. The dataset used contains features derived from astronomical observations, such as magnitudes in different filters (u, g, r, i, z), redshift, and observation metadata. The project aims to provide a clear and reproducible pipeline for this classification task.

##Features
Data Preprocessing: Includes handling missing values, encoding categorical labels (e.g., 'class'), feature scaling, and managing new/unknown data points.

Model Training: Demonstrates the training of multiple classification algorithms.

Model Evaluation: Provides comprehensive evaluation metrics including:

Accuracy Score

Classification Report (Precision, Recall, F1-score for each class)

Confusion Matrix visualization

Prediction Confidence visualization (histograms of maximum probabilities)

Individual Sample Inference: A utility function and examples to classify single, new astronomical objects and display their predicted class and probability distributions.

##Machine Learning Models Used
The project explores the performance of the following classification models:

Random Forest Classifier

Support Vector Machine (SVM)

XGBoost Classifier

Multi-layer Perceptron (Neural Network)

## Contents
Predicting Stars, Galaxies & Quasars with ML Model.ipynb: The main Jupyter Notebook containing the full code for data loading, preprocessing, model training, evaluation, and inference.

label_encoder.pkl: A serialized LabelEncoder object, used for transforming and inverse-transforming class labels.

scaler.pkl: A serialized StandardScaler object, used for scaling numerical features.

rf_model.pkl: The trained Random Forest Classifier model.

svm_model.pkl: The trained Support Vector Machine model.

xgb_sdss_classifier.pkl: The trained XGBoost Classifier model.

nn_model.pkl: The trained Neural Network model.

Skyserver_12_30_2019_4_49_58_PM.csv: (Or similar CSV files) The dataset used for training and evaluation.

.ipynb_checkpoints/: (Directory) Jupyter Notebook's auto-saved checkpoints.

##How to Run
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/YourUsername/YourRepositoryName.git
cd YourRepositoryName

(Replace YourUsername and YourRepositoryName with your actual GitHub username and repository name).

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

##Install dependencies:
Ensure you have all the necessary libraries installed. You can create a requirements.txt file by running pip freeze > requirements.txt after installing them, or manually create one with the following:

pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
If you used TensorFlow/Keras for the Neural Network:
tensorflow

Then install:

pip install -r requirements.txt

Launch Jupyter Notebook:

jupyter notebook

Open the Notebook: In the Jupyter interface, navigate to and open Predicting Stars, Galaxies & Quasars with ML Model.ipynb.

Run Cells: Execute the cells in the notebook sequentially to perform data preprocessing, load/train models, evaluate them, and perform sample predictions.

##Contact
For any questions or suggestions, please feel free to reach out.
