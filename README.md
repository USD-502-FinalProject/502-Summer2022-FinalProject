# Telecom Churn: Determining the impacts of customer churn on the telecommunications industry
This is a final project for ADS502: Applied Data Mining(https://github.com/USD-502-FinalProject/502-Summer2022-FinalProject)).

#### -- Project Status: [Completed]

## Project Intro
The purpose of this project is to present the process used to determine the best set of features to predict customer churn for a California based telecommunications company set in Q2-2022. (April-June 2022). Binary classifications models employed for this research project include Logisitc Regression, Decision Tree, and KNN. Finally, 5-Fold cross validation was carried out to evaluate each of the models for performance evaluation in predicting customer churn. The model performance evaluation metrics show that the set of features defined through our process are effective in predicting customer churn. With the added support from an additional three predicition models that are as effective if not more than those currently existing in the telecommunications field. 

### Team 3
* Connie Chow
* Christine Vu
* Vannesa Salazar


### Methods Used
* Data preprocessing
* Exploratory Data Analysis
* Data Visualization
* Predictive Modeling


### Technologies
* Python
* Pandas, jupyternotebook


## Project Description
-Customer churn is the measure of how many customers decide to no longer do business or maintain a subscription with a company during a specified timeframe. It is assumed that it is more costly for a company to consistently attempt to acquire new customers than to maintain a loyal customer. 
- A company would be eager to learn crucial insights on how to maintain long-term business. 
- By predicting if a customer is likely to churn, a business could prepare and implement strategies to reduce the churn process. 
- Various studies indicate it is ten times more costly to acquire new customers than it is to retain a customer. 
- Loyal customer's lifetime value tends to bring a much higher return than the cost of acquiring a new customer. 
- Throughout this project we will determine the most common attributes of a customer with the potential to churn and create a model to predit this potential. 
- Dataset provided by Kaggle 
- https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics
-The data is taken from a telecommunications company in California from Q2 2022.  


## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- presentation
- voice recordings

## Required Python Packages
* import pandas as pd
* import matplotlib
* import matplotlib.pyplot as plt
* import numpy as np
* from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
* import seaborn as sns
* from sklearn.experimental import enable_iterative_imputer
* from sklearn.impute import IterativeImputer
* from sklearn import linear_model
* from sklearn.impute import SimpleImputer
* from sklearn.neighbors import KNeighborsClassifier
* from collections import defaultdict
* from scipy import stats
* from sklearn.cluster import KMeans
* from pandas.api.types import CategoricalDtype
* %matplotlib inline
* from sklearn.ensemble import RandomForestClassifier
* import statsmodels.api as sm
* import statsmodels.tools.tools as stattools
* from sklearn.metrics import confusion_matrix
* from sklearn.tree import DecisionTreeClassifier, export_graphviz
* from sklearn.tree import plot_tree
* import random
* from sklearn.model_selection import train_test_split
* from sklearn.naive_bayes import MultinomialNB
* from sklearn.linear_model import LogisticRegression
* from sklearn.model_selection import KFold, cross_val_score
* from sklearn import tree

## Getting Started

1. Clone this repo using raw data.
2. add code and push new code into repo. To ensure cohesive code make sure to run all cells prior to upload. 
3. Use ###### flags for output

## Featured Notebooks/Analysis/Deliverables
* [Presentation slides ](https://docs.google.com/presentation/d/14RM_kKek7yXIJMSbz1JzJaUCNy4UuMgv5izydzaegHs/edit)


## Data Pre-Processing
* SimpleImputer class was used to impute missing column values on a case-by-case basis
* Boxplots were used to detect outliers - Number of Referrals, Avg Monthly GB Download, Total Refunds, Total Long Distance Charges, Total Revenue
* Dataset was rebalanced for equal number of Churn and Not Churned customers


## Exploratory Data Analysis

Univariate Analysis
* pandas-profiling package used to generate a report containing general statistics for each attribute in dataset, see link [here](https://github.com/USD-502-FinalProject/502-Summer2022-FinalProject/blob/main/Telecom%20Customer%20Churn%20Data%20-%20Univariate%20Analysis.html)

Multivariate Analysis



## Features Selection
* Refer to chi-square test in this file [here](https://github.com/USD-502-FinalProject/502-Summer2022-FinalProject/blob/main/chisquare-test2.py)
* Pairwise Correlation
* Correlation to Target
* Test for Column Variance


## Modeling & Evaluation
* Logistic Regression
* Decision Tree
* Naïve Bayes
* Random Forest
* K-means
* CART
* C5.0
