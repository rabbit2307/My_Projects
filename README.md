# Predicting Stars, Galaxies & Quasars with Machine Learning

This project builds a supervised machine learning pipeline to classify astronomical sources from the Sloan Digital Sky Survey (SDSS) as Stars, Galaxies, or Quasars based on photometric features. It demonstrates data-intensive astronomy analysis suitable for large surveys, with methods relevant to modern astronomical classification tasks.

## Project Goals
-Automate the classification of SDSS photometric sources into Star, Galaxy, or Quasar classes.

-Explore the features and distributions of the dataset.

-Train multiple machine learning models.

-Evaluate and compare their performance.

-Provide a simple deployment approach for practical classification.

## Dataset
Source: SDSS photometric catalog
Publicly available CSV

## Key Steps
#### Data Cleaning

-Dropped irrelevant columns (objid, specobjid).

-Handled class label encoding.

#### Exploratory Data Analysis (EDA)

-Visualized class distribution.

-Generated pairplots of magnitudes.

-Correlation heatmap.

#### Feature Engineering

-Derived color indices (planned).

-Scaled features with StandardScaler.

#### Model Training

-Decision Tree

-Logistic Regression

-K-Nearest Neighbors

-Random Forest (with hyperparameter tuning)

-XGBoost

-SVM

-Neural Network (MLP)

#### Model Evaluation

-Classification reports.

-Confusion matrices.

-Comparison of accuracy scores.

#### Hyperparameter Tuning

-Used GridSearchCV to optimize model parameters.

-Compared best results across models.

#### Deployment Example

-Built a Streamlit app to predict class from user input.

-Demonstrates real-time classification using trained model.

## Tools & Libraries
Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-Learn

XGBoost

Streamlit

Astroquery (for related data access)

SNooPy & Spextractor (for related SN work, not in this notebook but part of research background)

## Results
Best accuracy achieved: ~98.48% with Random Forest on test data.

Model comparison plot included.

Streamlit app for practical use.

## How to Use
Clone the repository.

Install dependencies (requirements.txt or manual).

Run the Jupyter notebook to see full analysis.

Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
## Why This Matters
This project simulates a real data-intensive astronomy workflow:

Works with large-scale survey data.

Demonstrates classification essential for catalogs like SDSS, LSST.

Shows end-to-end pipeline development from raw data to deployable app.

## Author
[Your Name Here]

Undergraduate student passionate about data-intensive astronomy.

## Acknowledgements
Sloan Digital Sky Survey (SDSS) for the dataset.

Brown Physics AI Winter School for ML training.

Guidance from Dr. Syeda Lammim Ahad, Dr. Andy Howell, Dr. Syed Ashraf Uddin Shuvo.

## License
For academic demonstration purposes.

