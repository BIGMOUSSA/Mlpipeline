import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import zipfile

## Loading dataset

rawData = pd.read_csv("data/raw/university_enrollment_2306.csv", index_col=0)
print(rawData.head())

# making pre_score on the right dtype,
rawData["pre_score"] = pd.to_numeric(rawData["pre_score"], errors='coerce')

# for each column replace the missing values by the given value

imp_dict ={"pre_score" : 0,
           "post_score" : 0,
           "pre_requirement": "None",
           "departement" : "unknown",
           "course_type" : "classroom",
           "year" : 2011}

#imputation
#for var, value in imp_dict.items():
rawData.fillna(value = imp_dict, inplace = True)

# let's select relevant columns

## lets clean the variable departemnt
rawData["department"] = rawData["department"].str.replace("Mathematics", "Math")

relevant_feat = ["pre_requirement", "department", "course_type"]

CleanData = rawData[relevant_feat]

one_hot = pd.get_dummies(CleanData, columns=relevant_feat)

cleaned_data = pd.concat([rawData["enrollment_count"],one_hot], axis = 1)

print(cleaned_data.info())

# let's split the data into train and test using skitlearn

train_data, test_data = train_test_split(cleaned_data, shuffle=True, test_size= 0.3)

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)


# Create a zipped folder and add the CSV files to it
with zipfile.ZipFile("data/cleaned", 'w') as zipf:
    zipf.write('train.csv')
    zipf.write('test.csv')

# Remove the temporary CSV files
import os
os.remove('train.csv')
os.remove('test.csv')




