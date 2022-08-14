import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sns as sns
from scipy import stats
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dython import nominal

df = pd.read_csv('https://raw.githubusercontent.com/USD-502-FinalProject/502-Summer2022-FinalProject/main/telecom_customer_churn.csv', sep=',', na_values = [""])


# Loop through each row of dataframe, assigning value for 'Customer Status'
# as either 0 or 1
for i, row in df.iterrows():

    val = 0  # default Stayed

    # Joined or Stayed
    if row['Customer Status'] == 'Stayed' or row['Customer Status'] == 'Joined':
        val = 0
    else:  # Churned
        val = 1

    df.at[i,'Customer Status'] = val


# Convert 'Customer Status' column into int/numeric
df['Customer Status'] = pd.to_numeric(df['Customer Status'])


def convertTuple(tup):
        # initialize an empty string
    str = ''
    for item in tup:
        str = str + ', ' + item
    return str


# Test of Independence: We have two categorical variable. We want to
# check if there is a relationship between these two.
# If two features are highly correlated, we can drop one of them.
#########################################################################################
# Exploratory Data Analysis - p-value chi-square test
#########################################################################################
# Need to take out normalized data?

# We will run chi-square analysis on every combination of
# predictor variables in the dataframe
# All results are printed to a text file.  We will use this
# in conjunction with the correlation matrix and basic EDA
# to determine the predictor variables used in feature selection.

# Create Contingency Table
# Null hypothesis assumes that the observed frequencies of these
# categorical variables is equal to the expected frequencies

# Come up with all possible 2 variable combinations for contingency
# table, drop any duplicate pairs

# All possible pairs in List
# Using combinations()
# https://stackoverflow.com/questions/18859430/how-do-i-get-the-total-number-of-unique-pairs-of-a-set-in-the-database
from itertools import combinations
#df = df.drop('Customer Status', axis=1)
test_list = df.columns
res = list(combinations(test_list, 2))
uniques = set(res)
result = list(uniques)
#result = result[:10]


import sys

f = open("chisquare-test-result2.txt", "w")
f.write("Chi-Square Test Results:")
f.write("This file lists the pairs of variables that were found to be highly correlated.  Use this to choose one to discard in feature selection process.")

# Then loop through each and print to file if they are strongly correlated
for entry in result:
	contingency = pd.crosstab(df[entry[0]], df[entry[1]])

# Chi-square Test predicts the expected frequencies of the values
# in the contingency table then determines if they match the observed
# frequencies.  The result is a test statistic that can be used to
# accept or reject the null hypothesis.
	stat, p, dof, expected = chi2_contingency(contingency)

# Test 1 and Test 2 below should have same result
# Test 1: interpret test-statistic
	#prob = 0.95
	#critical = chi2.ppf(prob, dof)
	#if abs(stat) >= critical:
		#f.write("\ndependent - test-statistic: \n")
		#f.write(convertTuple(entry))
		#f.write('Dependent (reject H0)\n\n')
#	else:
#		print('Independent (fail to reject H0)')


	# Test 2: interpret p-value
	prob = 0.95
	alpha = 1.0 - prob
	if p <= alpha:
		f.write("\ndependent: Interpret p-value: \n")
		f.write(convertTuple(entry))
		#f.write('Dependent (reject H0)\n\n')
	#else:
	#	f.write('Independent (fail to reject H0)')


f.close()